"""Train a YOLO detector on the tiled xView data and log the run to MLflow.

Usage:
    uv run python -m aerial_detect.train --smoke        # 1 epoch on 5% of data: pipeline check
    uv run python -m aerial_detect.train --epochs 30    # baseline
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from aerial_detect.groups import GROUP_NAMES

TILES = Path("data/tiles")
RUNS = Path("runs")
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "aerial-detect"


def dataset_config(tiles_dir: Path) -> dict[str, object]:
    """Ultralytics dataset config. The path must be absolute, or Ultralytics resolves it
    against its own datasets folder instead of this project."""
    return {
        "path": str(tiles_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(GROUP_NAMES)),
    }


def data_version(lock_file: Path = Path("dvc.lock"), out: str = "data/tiles") -> str:
    """The DVC hash of the tiled dataset, read from dvc.lock."""
    lock = yaml.safe_load(lock_file.read_text())
    for stage in lock["stages"].values():
        for item in stage.get("outs", []):
            if item["path"] == out:
                return str(item["md5"])
    raise KeyError(f"{out} not found in {lock_file}")


def git_state() -> tuple[str, bool]:
    """Current commit, and whether there are uncommitted changes."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout
    return commit, bool(status.strip())


def clean_metric_name(name: str) -> str:
    """'metrics/mAP50-95(B)' -> 'metrics_mAP50-95' (MLflow rejects some characters)."""
    name = name.strip().replace("(B)", "").replace("/", "_")
    return re.sub(r"[^A-Za-z0-9_.\- ]", "", name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fraction", type=float, default=1.0, help="share of train tiles")
    parser.add_argument("--smoke", action="store_true", help="1 epoch on 5%% of the data")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    if args.smoke:
        args.epochs, args.fraction = 1, 0.05

    import mlflow  # heavy imports here, so CI can test this file without them
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False})  # this script does its own MLflow logging

    RUNS.mkdir(exist_ok=True)
    data_yaml = RUNS / "dataset.yaml"
    data_yaml.write_text(yaml.safe_dump(dataset_config(TILES), sort_keys=False))

    commit, dirty = git_state()
    suffix = "-smoke" if args.smoke else ""
    run_name = args.name or f"{Path(args.model).stem}-e{args.epochs}{suffix}"

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": args.model,
                "epochs": args.epochs,
                "batch": args.batch,
                "imgsz": args.imgsz,
                "fraction": args.fraction,
            }
        )
        mlflow.set_tags(
            {"git_commit": commit, "git_dirty": str(dirty), "data_version": data_version()}
        )

        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            fraction=args.fraction,
            device=0,
            seed=0,
            deterministic=True,
            max_det=1000,  # dense tiles can hold hundreds of buildings
            project=str((RUNS / "train").resolve()),
            name=run_name,
            exist_ok=True,
        )
        trainer = model.trainer
        assert trainer is not None, "training did not start"
        save_dir = Path(trainer.save_dir)

        # Per-epoch curves
        history = pd.read_csv(save_dir / "results.csv")
        for _, row in history.iterrows():
            values = {
                clean_metric_name(k): float(v) for k, v in row.items() if k.strip() != "epoch"
            }
            mlflow.log_metrics(values, step=int(row["epoch"]))
            mlflow.log_metrics(values, step=int(row["epoch"]))

        # Final validation, overall and per class (test split stays untouched)
        metrics = model.val(
            data=str(data_yaml),
            split="val",
            imgsz=args.imgsz,
            batch=max(1, args.batch // 2),  # dense tiles: smaller batch avoids GPU memory errors
            max_det=1000,
            device=0,
            project=str((RUNS / "val").resolve()),
            name=run_name,
            exist_ok=True,
        )
        mlflow.log_metrics(
            {"val_mAP50": float(metrics.box.map50), "val_mAP50_95": float(metrics.box.map)}
        )
        for idx, ap50, ap in zip(
            metrics.box.ap_class_index, metrics.box.ap50, metrics.box.ap, strict=True
        ):
            name = GROUP_NAMES[int(idx)]
            mlflow.log_metrics({f"val_AP50_{name}": float(ap50), f"val_AP50_95_{name}": float(ap)})

        mlflow.log_artifacts(str(save_dir), artifact_path="train")
        print(f"Done. val mAP50 = {metrics.box.map50:.3f}, run '{run_name}' logged to MLflow.")


if __name__ == "__main__":
    main()
