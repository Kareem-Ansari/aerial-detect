import io
from collections.abc import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from aerial_detect.api import create_app
from aerial_detect.inference import TileDetections


def fake_predictor(tiles: list[np.ndarray]) -> list[TileDetections]:
    """One small vehicle at tile pixels 100-140 in every tile."""
    box = np.array([[100.0, 100.0, 140.0, 140.0]])
    return [(box, np.array([0.9]), np.array([1])) for _ in tiles]


def geotiff_bytes(w: int = 1000, h: int = 1000) -> bytes:
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=w,
            height=h,
            count=3,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(10.0, 50.0, 1e-5, 1e-5),
        ) as dst:
            dst.write(np.full((3, h, w), 100, dtype=np.uint8))
        return bytes(mem.read())


def png_bytes(w: int = 1000, h: int = 1000) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(fake_predictor)) as c:
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_geotiff_returns_pixels_and_lonlat(client: TestClient) -> None:
    r = client.post("/predict", files={"file": ("img.tif", geotiff_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert body["georeferenced"] is True
    assert body["n_detections"] == 4  # one per tile, none overlapping
    assert body["counts"]["small_vehicle"] == 4
    first = next(d for d in body["detections"] if d["bbox"] == [100.0, 100.0, 140.0, 140.0])
    assert first["lon"] == pytest.approx(10.0012)  # 10 + 120 px * 1e-5
    assert first["lat"] == pytest.approx(49.9988)  # 50 - 120 px * 1e-5


def test_predict_png_has_no_lonlat(client: TestClient) -> None:
    body = client.post("/predict", files={"file": ("img.png", png_bytes())}).json()
    assert body["georeferenced"] is False
    assert all(d["lon"] is None for d in body["detections"])


def test_min_score_filters(client: TestClient) -> None:
    files = {"file": ("img.tif", geotiff_bytes())}
    body = client.post("/predict", params={"min_score": 0.95}, files=files).json()
    assert body["n_detections"] == 0


def test_unreadable_file_is_400(client: TestClient) -> None:
    r = client.post("/predict", files={"file": ("x.tif", b"not an image")})
    assert r.status_code == 400


def test_geojson_for_georeferenced_image(client: TestClient) -> None:
    r = client.post("/predict/geojson", files={"file": ("img.tif", geotiff_bytes())})
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "FeatureCollection"
    assert len(body["features"]) == 4
    assert len(body["features"][0]["geometry"]["coordinates"][0]) == 5  # closed ring


def test_geojson_needs_georeference(client: TestClient) -> None:
    r = client.post("/predict/geojson", files={"file": ("img.png", png_bytes())})
    assert r.status_code == 422
