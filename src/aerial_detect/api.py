"""HTTP API: upload a satellite image, get the detected objects back as JSON or GeoJSON.

Run locally:
    uv run uvicorn aerial_detect.api:app --port 8000
Then open http://127.0.0.1:8000/docs
"""

import os
import threading
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import MemoryFile
from rasterio.warp import transform as warp_transform

from aerial_detect.groups import GROUP_NAMES
from aerial_detect.inference import TilePredictor, predict_image, yolo_predictor

DEFAULT_WEIGHTS = "runs/train/yolo11s-e30/weights/best.pt"
MinScore = Annotated[float, Query(ge=0, le=1, description="Drop detections below this score")]


class Detection(BaseModel):
    class_id: int
    class_name: str
    score: float
    bbox: list[float]  # xmin, ymin, xmax, ymax in image pixels
    lon: float | None = None  # box centre, when the image is georeferenced
    lat: float | None = None


class PredictResponse(BaseModel):
    width: int
    height: int
    georeferenced: bool
    n_detections: int
    counts: dict[str, int]
    inference_ms: float
    detections: list[Detection]


@dataclass
class Prediction:
    width: int
    height: int
    transform: Any
    crs: CRS | None
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    ms: float


def read_image(data: bytes) -> tuple[np.ndarray, Any, CRS | None]:
    """Decode GeoTIFF, PNG or JPEG bytes to H x W x 3 uint8 pixels, plus transform and CRS."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(data) as mem, mem.open() as src:
            img = np.transpose(src.read(), (1, 2, 0))
            crs, transform = src.crs, src.transform
    if img.dtype != np.uint8:
        raise ValueError(f"expected 8-bit pixels, got {img.dtype}")
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.ascontiguousarray(img[:, :, :3]), transform, crs


def to_lonlat(
    transform: Any, crs: CRS, xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates -> longitude/latitude (EPSG:4326)."""
    gx, gy = transform * (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    if crs.to_epsg() != 4326:
        gx, gy = warp_transform(crs, CRS.from_epsg(4326), list(gx), list(gy))
    return np.asarray(gx, dtype=float), np.asarray(gy, dtype=float)


def create_app(predictor: TilePredictor | None = None) -> FastAPI:
    """Build the app. Tests pass a fake predictor; in production the model loads at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        weights = os.environ.get("MODEL_PATH", DEFAULT_WEIGHTS)
        app.state.predictor = predictor or yolo_predictor(weights)
        yield

    app = FastAPI(title="aerial-detect", version="0.1.0", lifespan=lifespan)
    model_lock = threading.Lock()  # one request uses the model at a time

    def run(file: UploadFile, min_score: float) -> Prediction:
        try:
            img, transform, crs = read_image(file.file.read())
        except (RasterioIOError, ValueError) as err:
            raise HTTPException(status_code=400, detail=f"Could not read image: {err}") from err
        start = time.perf_counter()
        with model_lock:
            boxes, scores, classes = predict_image(img, app.state.predictor)
        ms = (time.perf_counter() - start) * 1000
        keep = scores >= min_score
        h, w = img.shape[:2]
        return Prediction(w, h, transform, crs, boxes[keep], scores[keep], classes[keep], ms)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(file: UploadFile, min_score: MinScore = 0.25) -> PredictResponse:
        p = run(file, min_score)
        lons: list[float | None] = [None] * len(p.boxes)
        lats: list[float | None] = [None] * len(p.boxes)
        if p.crs is not None and len(p.boxes):
            cx = (p.boxes[:, 0] + p.boxes[:, 2]) / 2
            cy = (p.boxes[:, 1] + p.boxes[:, 3]) / 2
            gx, gy = to_lonlat(p.transform, p.crs, cx, cy)
            lons, lats = [float(v) for v in gx], [float(v) for v in gy]
        detections = [
            Detection(
                class_id=int(c),
                class_name=GROUP_NAMES[int(c)],
                score=round(float(s), 4),
                bbox=[round(float(v), 1) for v in b],
                lon=lons[i],
                lat=lats[i],
            )
            for i, (b, s, c) in enumerate(zip(p.boxes, p.scores, p.classes, strict=True))
        ]
        counts = {name: int((p.classes == i).sum()) for i, name in enumerate(GROUP_NAMES)}
        return PredictResponse(
            width=p.width,
            height=p.height,
            georeferenced=p.crs is not None,
            n_detections=len(detections),
            counts=counts,
            inference_ms=round(p.ms, 1),
            detections=detections,
        )

    @app.post("/predict/geojson")
    def predict_geojson(file: UploadFile, min_score: MinScore = 0.25) -> JSONResponse:
        p = run(file, min_score)
        if p.crs is None:
            raise HTTPException(status_code=422, detail="Image has no georeference; use /predict")
        features = []
        for b, s, c in zip(p.boxes, p.scores, p.classes, strict=True):
            xs = np.array([b[0], b[2], b[2], b[0], b[0]])
            ys = np.array([b[1], b[1], b[3], b[3], b[1]])
            gx, gy = to_lonlat(p.transform, p.crs, xs, ys)
            ring = [[float(x), float(y)] for x, y in zip(gx, gy, strict=True)]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"class_name": GROUP_NAMES[int(c)], "score": round(float(s), 4)},
                }
            )
        return JSONResponse(
            {"type": "FeatureCollection", "features": features}, media_type="application/geo+json"
        )

    return app


app = create_app()
