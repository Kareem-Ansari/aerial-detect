"""Run the tile detector over a whole image and merge duplicate detections across tiles."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aerial_detect.prepare import OVERLAP, TILE, crop_tile
from aerial_detect.tiling import tile_windows

# One tile's detections: boxes (N x 4, pixels), scores (N,), class ids (N,)
TileDetections = tuple[np.ndarray, np.ndarray, np.ndarray]
TilePredictor = Callable[[list[np.ndarray]], list[TileDetections]]


def empty_detections() -> TileDetections:
    return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)


def nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5, ios_thr: float = 0.7
) -> np.ndarray:
    """Indices of boxes to keep, highest score first.

    A box is dropped if it overlaps a higher-scoring box by IoU above iou_thr, or if most of
    the smaller box lies inside it (intersection over smaller area above ios_thr). The second
    rule catches a partial detection at a tile edge sitting inside the full detection."""
    order = np.argsort(-scores, kind="stable")
    x0, y0, x1, y1 = boxes.T
    areas = (x1 - x0) * (y1 - y0)
    keep: list[int] = []
    while order.size > 0:
        i, rest = order[0], order[1:]
        keep.append(int(i))
        w = np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
        h = np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        ios = inter / (np.minimum(areas[i], areas[rest]) + 1e-9)
        order = rest[(iou <= iou_thr) & (ios <= ios_thr)]
    return np.array(keep, dtype=int)


def merge_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_thr: float = 0.5,
    ios_thr: float = 0.7,
) -> TileDetections:
    """Run NMS separately per class, then sort everything by score."""
    kept: list[np.ndarray] = []
    for c in np.unique(classes):
        idx = np.flatnonzero(classes == c)
        kept.append(idx[nms(boxes[idx], scores[idx], iou_thr, ios_thr)])
    if not kept:
        return empty_detections()
    sel = np.concatenate(kept)
    sel = sel[np.argsort(-scores[sel], kind="stable")]
    return boxes[sel], scores[sel], classes[sel]


def predict_image(
    img: np.ndarray, predictor: TilePredictor, tile: int = TILE, overlap: float = OVERLAP
) -> TileDetections:
    """Detect objects in a full H x W x 3 RGB image: tile it, predict every tile, shift boxes
    back to image pixels, clip them to the image, and merge duplicates from overlapping tiles."""
    height, width = img.shape[:2]
    windows = tile_windows(width, height, tile, overlap)
    results = predictor([crop_tile(img, w) for w in windows])

    boxes_l, scores_l, classes_l = [], [], []
    for (x0, y0, _, _), (b, s, c) in zip(windows, results, strict=True):
        if len(b) == 0:
            continue
        boxes_l.append(np.asarray(b, dtype=float) + np.array([x0, y0, x0, y0], dtype=float))
        scores_l.append(np.asarray(s, dtype=float))
        classes_l.append(np.asarray(c, dtype=int))
    if not boxes_l:
        return empty_detections()

    boxes = np.concatenate(boxes_l)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)  # padded edge tiles can overshoot
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return merge_detections(boxes, np.concatenate(scores_l), np.concatenate(classes_l))


def yolo_predictor(weights: str, conf: float = 0.25, batch: int = 8) -> TilePredictor:
    """Wrap an Ultralytics model as a TilePredictor. Imported lazily: CI has no Ultralytics."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def predict(tiles: list[np.ndarray]) -> list[TileDetections]:
        out: list[TileDetections] = []
        for i in range(0, len(tiles), batch):
            # Ultralytics treats NumPy images as BGR (OpenCV order); our tiles are RGB.
            bgr = [np.ascontiguousarray(t[:, :, ::-1]) for t in tiles[i : i + batch]]
            for r in model.predict(bgr, imgsz=TILE, conf=conf, max_det=1000, verbose=False):
                data = np.zeros((0, 6)) if r.boxes is None else r.boxes.data.cpu().numpy()  # type: ignore
                out.append((data[:, :4], data[:, 4], data[:, 5].astype(int)))
        return out

    return predict
