"""Cut large images into overlapping fixed-size tiles and move box labels into tile coordinates."""

import numpy as np

Window = tuple[int, int, int, int]  # (x0, y0, x1, y1) in image pixels


def _starts(length: int, tile: int, stride: int) -> list[int]:
    """Start positions along one axis, with the last tile ending exactly at the edge.

    If the regular tiles leave only a small gap (no bigger than the overlap), the last
    tile is shifted to the edge instead of adding a near-duplicate tile."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    gap = length - (starts[-1] + tile)
    if gap == 0:
        return starts
    if gap <= tile - stride and len(starts) > 1:
        starts[-1] = length - tile  # shift: the previous tile still overlaps it
    else:
        starts.append(length - tile)
    return starts


def tile_windows(width: int, height: int, tile: int = 640, overlap: float = 0.2) -> list[Window]:
    """Return tile windows that cover the whole image. Every window is tile x tile."""
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if tile <= 0:
        raise ValueError(f"tile must be positive, got {tile}")
    stride = max(1, int(tile * (1 - overlap)))
    xs = _starts(width, tile, stride)
    ys = _starts(height, tile, stride)
    return [(x, y, x + tile, y + tile) for y in ys for x in xs]


def clip_boxes(
    boxes: np.ndarray, window: Window, min_visible: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Shift boxes (N x 4: xmin, ymin, xmax, ymax) into window coordinates, clip them to the
    window, and drop boxes with less than `min_visible` of their area inside.

    Returns (clipped_boxes, kept_indices)."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x0, y0, x1, y1 = window

    # Overlap rectangle between each box and the window
    ix0 = np.maximum(boxes[:, 0], x0)
    iy0 = np.maximum(boxes[:, 1], y0)
    ix1 = np.minimum(boxes[:, 2], x1)
    iy1 = np.minimum(boxes[:, 3], y1)

    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    keep = np.flatnonzero((area > 0) & (inter >= min_visible * area))
    clipped = np.stack([ix0, iy0, ix1, iy1], axis=1)[keep]
    clipped -= np.array([x0, y0, x0, y0], dtype=float)
    return clipped, keep
