from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from aerial_detect.prepare import (
    ImageJob,
    crop_tile,
    keep_empty_tile,
    labels_table,
    process_image,
    to_yolo,
)


def _write_tif(path: Path, width: int, height: int) -> None:
    data = np.full((3, height, width), 128, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1e-5, 1e-5),
    ) as dst:
        dst.write(data)


def test_labels_table_keeps_only_grouped_classes() -> None:
    raw = pd.DataFrame(
        {
            "image_id": ["a.tif"] * 4,
            # Building, Small Car, Shipping Container (left out), unused id
            "type_id": [73, 18, 91, 75],
            "bounds_imcoords": ["1,2,3,4", "5,6,7,8", "9,10,11,12", "13,14,15,16"],
        }
    )
    out = labels_table(raw)
    assert out["group"].tolist() == [4, 1]  # building, small_vehicle
    assert out[["xmin", "ymin", "xmax", "ymax"]].iloc[1].tolist() == [5, 6, 7, 8]


def test_to_yolo_normalizes_by_tile_size() -> None:
    line = to_yolo(np.array([[64.0, 64.0, 192.0, 128.0]]), np.array([2]), tile=640)
    assert line == "2 0.200000 0.150000 0.200000 0.100000\n"


def test_to_yolo_empty_is_empty_string() -> None:
    assert to_yolo(np.zeros((0, 4)), np.zeros(0, dtype=int)) == ""


def test_crop_tile_pads_past_the_edge() -> None:
    img = np.ones((200, 300, 3), dtype=np.uint8)
    tile = crop_tile(img, (0, 0, 640, 640))
    assert tile.shape == (640, 640, 3)
    assert tile[:200, :300].all() and not tile[200:, :].any() and not tile[:, 300:].any()


def test_empty_tiles_kept_outside_train_and_sampled_repeatably_in_train() -> None:
    assert keep_empty_tile("x_0_0", "test")
    assert keep_empty_tile("x_0_0", "holdout")
    ids = [f"img_{i}_0" for i in range(2000)]
    kept = [t for t in ids if keep_empty_tile(t, "train")]
    assert 0.07 < len(kept) / len(ids) < 0.13  # about 10%
    assert kept == [t for t in ids if keep_empty_tile(t, "train")]  # same result every run


def test_process_image_writes_tiles_labels_and_manifest_rows(tmp_path: Path) -> None:
    tif = tmp_path / "img.tif"
    _write_tif(tif, 1000, 1000)
    job = ImageJob(
        image_path=tif,
        split="test",
        site=3,
        boxes=np.array([[400.0, 100.0, 440.0, 140.0]]),
        groups=np.array([4]),
        out_dir=tmp_path / "out",
    )
    rows = process_image(job)

    assert len(rows) == 4  # 1000 px -> 2 x 2 tiles; all kept outside train
    tiles = sorted((tmp_path / "out" / "images" / "test").glob("*.jpg"))
    assert len(tiles) == 4
    assert Image.open(tiles[0]).size == (640, 640)

    # The box (x 400-440) falls in tiles starting at x=0 and x=360
    assert sum(int(r["n_objects"]) for r in rows) == 2  # type: ignore[call-overload]
    label = (tmp_path / "out" / "labels" / "test" / "img_360_0.txt").read_text()
    assert label == "4 0.093750 0.187500 0.062500 0.062500\n"  # x 40-80, y 100-140 in the tile
