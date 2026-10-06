import numpy as np
import pandas as pd
import pytest

from aerial_detect.split import assign_sites, make_split, split_sites


def test_nearby_points_share_a_site_far_points_dont() -> None:
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0, 0.09, 5.0])  # ~10 km and ~556 km from the first
    sites = assign_sites(lat, lon, radius_km=50)
    assert sites[0] == sites[1]
    assert sites[0] != sites[2]


def test_split_hits_target_image_shares() -> None:
    sizes = {i: 10 for i in range(100)}  # 1,000 images
    assignment = split_sites(sizes)
    totals = {s: 0 for s in ("train", "val", "test")}
    for site, split in assignment.items():
        totals[split] += sizes[site]
    assert totals == {"train": 700, "val": 150, "test": 150}


def test_every_site_assigned_once() -> None:
    sizes = {i: i + 1 for i in range(30)}
    assignment = split_sites(sizes)
    assert set(assignment) == set(sizes)
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_biggest_site_goes_to_train() -> None:
    sizes = {0: 144, **{i: 10 for i in range(1, 41)}}
    assert split_sites(sizes)[0] == "train"


def test_same_seed_same_split() -> None:
    sizes = {i: (i % 7) + 1 for i in range(40)}
    assert split_sites(sizes, seed=3) == split_sites(sizes, seed=3)


@pytest.mark.parametrize("fractions", [(0.7, 0.2, 0.2), (1.0, 0.0, 0.0), (0.5, 0.5, -0.0)])
def test_bad_fractions_raise(fractions: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        split_sites({0: 10}, fractions)


def _cluster(lat: float, lon: float, n: int, prefix: str) -> list[dict[str, object]]:
    return [{"image_id": f"{prefix}{i}.tif", "lat": lat + i * 1e-3, "lon": lon} for i in range(n)]


def test_holdout_is_south_america_minus_its_largest_site() -> None:
    rows = (
        _cluster(-23.0, -70.0, 20, "big_sa")  # largest South American site: stays in training
        + _cluster(-34.0, -58.0, 5, "small_sa")  # smaller South American site: held out
        + [r for k in range(10) for r in _cluster(40.0, k * 15.0, 8, f"other{k}_")]
    )
    out = make_split(pd.DataFrame(rows))
    assert set(out.loc[out.image_id.str.startswith("small_sa"), "split"]) == {"holdout"}
    assert "holdout" not in set(out.loc[out.image_id.str.startswith("big_sa"), "split"])
    assert (out.groupby("site")["split"].nunique() == 1).all()  # no site spans two splits
