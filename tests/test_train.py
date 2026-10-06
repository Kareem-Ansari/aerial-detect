from pathlib import Path

import yaml

from aerial_detect.train import clean_metric_name, data_version, dataset_config


def test_dataset_config_has_absolute_path_and_six_names(tmp_path: Path) -> None:
    cfg = dataset_config(tmp_path / "tiles")
    assert Path(str(cfg["path"])).is_absolute()
    assert cfg["names"] == {
        0: "aircraft",
        1: "small_vehicle",
        2: "large_vehicle",
        3: "ship",
        4: "building",
        5: "storage_tank",
    }


def test_data_version_reads_dvc_lock(tmp_path: Path) -> None:
    lock = {
        "schema": "2.0",
        "stages": {"prepare": {"outs": [{"path": "data/tiles", "md5": "abc123.dir"}]}},
    }
    lock_file = tmp_path / "dvc.lock"
    lock_file.write_text(yaml.safe_dump(lock))
    assert data_version(lock_file) == "abc123.dir"


def test_clean_metric_name() -> None:
    assert clean_metric_name("metrics/mAP50-95(B)") == "metrics_mAP50-95"
    assert clean_metric_name("  train/box_loss ") == "train_box_loss"
