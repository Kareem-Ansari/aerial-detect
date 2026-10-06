"""Group images into geographic sites and split them by whole site, so no site leaks."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0
SPLITS: tuple[str, str, str] = ("train", "val", "test")


def assign_sites(lat: np.ndarray, lon: np.ndarray, radius_km: float = 50.0) -> np.ndarray:
    """Cluster points into sites: points within radius_km of each other share a site id."""
    coords = np.radians(np.column_stack([lat, lon]))
    model = DBSCAN(
        eps=radius_km / EARTH_RADIUS_KM, min_samples=1, metric="haversine", algorithm="ball_tree"
    )
    return model.fit_predict(coords)


def holdout_sites(images: pd.DataFrame) -> set[int]:
    """Sites held out as a never-seen region: South America (lon < -30, lat < 0)
    except its largest site, which stays in training."""
    south_america = images[(images["lon"] < -30) & (images["lat"] < 0)]
    sizes = south_america.groupby("site").size()
    if sizes.empty:
        return set()
    largest = sizes.idxmax()
    return {int(s) for s in sizes.index if s != largest}


def split_sites(
    images_per_site: dict[int, int],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[int, str]:
    """Assign whole sites to train/val/test so each split gets close to its share of images.

    Greedy: place the biggest sites first, each into the split furthest below its target."""
    if len(fractions) != 3 or min(fractions) <= 0 or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must be 3 positive numbers summing to 1, got {fractions}")

    sites = list(images_per_site)
    np.random.default_rng(seed).shuffle(sites)  # random order for ties...
    sites.sort(key=lambda s: images_per_site[s], reverse=True)  # ...then biggest first (stable)

    total = sum(images_per_site.values())
    targets = [f * total for f in fractions]
    filled = [0.0, 0.0, 0.0]
    assignment: dict[int, str] = {}
    for site in sites:
        k = max(range(3), key=lambda i: targets[i] - filled[i])
        assignment[site] = SPLITS[k]
        filled[k] += images_per_site[site]
    return assignment


def make_split(
    images: pd.DataFrame,
    radius_km: float = 50.0,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> pd.DataFrame:
    """Add `site` and `split` (train/val/test/holdout) columns to a table of image_id, lat, lon."""
    out = images.copy()
    out["site"] = assign_sites(out["lat"].to_numpy(), out["lon"].to_numpy(), radius_km)
    held = holdout_sites(out)
    rest = out[~out["site"].isin(held)]
    sizes = {int(k): int(v) for k, v in rest.groupby("site").size().items()}
    mapping = split_sites(sizes, fractions, seed)
    out["split"] = out["site"].map(mapping).fillna("holdout")
    return out
