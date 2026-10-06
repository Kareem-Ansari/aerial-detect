"""Turn xView images and labels into 640 x 640 training tiles, YOLO label files and a manifest.

Usage:
    uv run python -m aerial_detect.prepare --limit 5 --clean   # quick trial on 5 images
    uv run python -m aerial_detect.prepare --clean             # all images
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from PIL import Image

from aerial_detect.classes import to_index
from aerial_detect.groups import GROUP_NAMES, to_group
from aerial_detect.tiling import Window, clip_boxes, tile_windows

TILE = 640
OVERLAP = 0.2
EMPTY_KEEP_RATE = 0.10  # share of object-free tiles kept in train (decision 007)
JPEG_QUALITY = 95  # decision 006
NODATA_MAX = 0.95  # skip empty tiles that are more than 95% no-data (decision 008)


@dataclass(frozen=True)
class ImageJob:
    """Everything one worker needs to tile one image."""

    image_path: Path
    split: str
    site: int
    boxes: np.ndarray  # N x 4: xmin, ymin, xmax, ymax in image pixels
    groups: np.ndarray  # N group indices (0-5)
    out_dir: Path


def labels_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw xView rows (image_id, type_id, bounds_imcoords) -> one row per kept object:
    image_id, group, xmin, ymin, xmax, ymax. Objects outside the 6 groups are dropped."""
    boxes = raw["bounds_imcoords"].str.split(",", expand=True).astype(float)
    out = pd.DataFrame(
        {
            "image_id": raw["image_id"].to_numpy(),
            "group": [to_group(to_index(int(t))) for t in raw["type_id"]],
            "xmin": boxes[0].to_numpy(),
            "ymin": boxes[1].to_numpy(),
            "xmax": boxes[2].to_numpy(),
            "ymax": boxes[3].to_numpy(),
        }
    )
    out = out[out["group"] >= 0]
    out = out.drop_duplicates(["image_id", "group", "xmin", "ymin", "xmax", "ymax"])
    return out.reset_index(drop=True)


def to_yolo(boxes: np.ndarray, groups: np.ndarray, tile: int = TILE) -> str:
    """YOLO label lines: 'group cx cy w h', all as fractions of the tile size."""
    lines = []
    for (x0, y0, x1, y1), g in zip(boxes, groups, strict=True):
        cx, cy = (x0 + x1) / 2 / tile, (y0 + y1) / 2 / tile
        w, h = (x1 - x0) / tile, (y1 - y0) / tile
        lines.append(f"{int(g)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def crop_tile(img: np.ndarray, window: Window) -> np.ndarray:
    """Cut a window out of an H x W x C image, zero-padding if it runs past the edge."""
    x0, y0, x1, y1 = window
    tile = np.zeros((y1 - y0, x1 - x0, img.shape[2]), dtype=img.dtype)
    part = img[y0:y1, x0:x1]
    tile[: part.shape[0], : part.shape[1]] = part
    return tile


def keep_empty_tile(tile_id: str, split: str, rate: float = EMPTY_KEEP_RATE, seed: int = 0) -> bool:
    """Keep every empty tile outside train; in train keep a repeatable random share."""
    if split != "train":
        return True
    return random.Random(f"{seed}:{tile_id}").random() < rate


def nodata_fraction(tile: np.ndarray) -> float:
    """Share of pixels that are exactly 0 in every band (the satellite's no-data fill)."""
    return float(np.all(tile == 0, axis=2).mean())


def process_image(job: ImageJob) -> list[dict[str, object]]:
    """Tile one image, write tile JPEGs and YOLO labels, return one manifest row per tile."""
    with rasterio.open(job.image_path) as src:
        img = np.transpose(src.read(), (1, 2, 0))[:, :, :3]  # bands-first -> H x W x 3
    if img.dtype != np.uint8:
        raise ValueError(f"{job.image_path.name}: expected uint8 pixels, got {img.dtype}")

    img_dir = job.out_dir / "images" / job.split
    lbl_dir = job.out_dir / "labels" / job.split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    height, width = img.shape[:2]
    rows: list[dict[str, object]] = []
    for window in tile_windows(width, height, TILE, OVERLAP):
        tile_boxes, keep = clip_boxes(job.boxes, window)
        tile_groups = job.groups[keep].astype(int)
        tile_id = f"{job.image_path.stem}_{window[0]}_{window[1]}"
        tile = crop_tile(img, window)
        nodata = nodata_fraction(tile)

        if len(keep) == 0:
            if nodata > NODATA_MAX:
                continue  # blank satellite border: nothing to learn or evaluate
            if not keep_empty_tile(tile_id, job.split):
                continue

        jpg = img_dir / f"{tile_id}.jpg"
        Image.fromarray(tile).save(jpg, quality=JPEG_QUALITY)
        (lbl_dir / f"{tile_id}.txt").write_text(to_yolo(tile_boxes, tile_groups))

        counts = np.bincount(tile_groups, minlength=len(GROUP_NAMES))
        rows.append(
            {
                "tile_id": tile_id,
                "split": job.split,
                "source_image": job.image_path.name,
                "site": job.site,
                "x0": window[0],
                "y0": window[1],
                "x1": window[2],
                "y1": window[3],
                "n_objects": int(len(keep)),
                **{f"n_{name}": int(c) for name, c in zip(GROUP_NAMES, counts, strict=True)},
                "nodata_frac": round(nodata, 4),
                "sha256": hashlib.sha256(jpg.read_bytes()).hexdigest(),
            }
        )
    return rows


def build_jobs(raw_dir: Path, splits_csv: Path, out_dir: Path, limit: int | None) -> list[ImageJob]:
    """Pair every image in the split file with its labels."""
    raw = pyogrio.read_dataframe(
        raw_dir / "xView_train.geojson",
        columns=["image_id", "type_id", "bounds_imcoords"],
        read_geometry=False,
    )
    by_image = dict(tuple(labels_table(raw).groupby("image_id")))
    splits = pd.read_csv(splits_csv)

    jobs = []
    for row in splits.itertuples(index=False):
        path = raw_dir / "train_images" / row.image_id
        if not path.exists():
            continue  # e.g. 1395.tif, missing from the release
        labels = by_image.get(row.image_id)
        if labels is None:
            boxes, groups = np.zeros((0, 4)), np.zeros(0, dtype=int)
        else:
            boxes = labels[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=float)
            groups = labels["group"].to_numpy(dtype=int)
        jobs.append(ImageJob(path, str(row.split), int(row.site), boxes, groups, out_dir))
    return jobs[:limit] if limit else jobs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", type=Path, default=Path("splits/image_splits.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/tiles"))
    parser.add_argument("--limit", type=int, default=None, help="only the first N images")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--clean", action="store_true", help="delete --out first")
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.raw, args.splits, args.out, args.limit)
    print(f"Tiling {len(jobs)} images with {args.workers} workers")

    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_image, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} images done, {len(rows):,} tiles")

    manifest = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)
    manifest.to_parquet(args.out / "manifest.parquet", index=False)
    summary = manifest.groupby("split").agg(tiles=("tile_id", "size"), objects=("n_objects", "sum"))
    print(summary)


if __name__ == "__main__":
    main()
