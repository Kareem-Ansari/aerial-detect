# Data

## Source
xView 2018 (DIUx): satellite imagery at about 0.3 m per pixel, 60 object classes.
License: CC BY-NC-SA 4.0 (non-commercial use with attribution).
Not included in this repo. Download manually after registering at challenge.xviewdataset.org.

## Files used
- `train_images.zip`: labelled GeoTIFFs
- `train_labels.zip`: `xView_train.geojson` (one feature per object)
- `val_images.zip`: unlabelled; used only for demo inference
- Verify your download: from the download folder, run `sha256sum -c docs/xview_checksums.sha256`
- Note: `train_images.zip` fails with Ubuntu's `unzip` ("possible zip bomb"). Extract with `7z x train_images.zip`, then delete the `__MACOSX` folder it contains.
- 846 training images extracted (23 GB); `1395.tif` is missing from the release.

## Layout
- `data/raw/train_images/*.tif`
- `data/raw/xView_train.geojson`

## What the data looks like

Findings from `notebooks/01_explore.ipynb`.

### Size
- 601,937 objects across 847 labelled images; 846 image files.
- `1395.tif` is missing, so its 131 objects are skipped.
- 79 objects have a `type_id` that is not one of the 60 classes; dropped.

### Class imbalance
- Building (316,795; 53%) and Small Car (211,664; 35%) make up 88% of all objects.
- The third-largest class, Truck, has 12,189 (2%). Many classes have only a few hundred examples.
- **Implication:** report mAP per class, not only overall. A model that only finds buildings and cars would still score well overall.

### Object size
- Median box is 25 x 23 px; 5% are 9 px or smaller; 0.9% are under 10 px on their longest side.
- Images are about 3,300 x 2,900 px, so objects are tiny relative to the image.
- **Implication:** tile at full resolution instead of downscaling. Shrinking a whole image would turn a 25 px car into about 5 px.

### Image format
- 3-band RGB GeoTIFFs, all in EPSG:4326, about 3 x 10^-6 degrees per pixel (about 0.3 m).

### Label quality
9,937 boxes (1.65%) fail at least one basic check. A box can fail more than one:

| Problem | Boxes |
| --- | --- |
| Starts before the image edge (negative coordinates) | 5,058 |
| Extends past the right edge | 2,615 |
| Extends past the bottom edge | 2,368 |
| Zero or negative width/height | 9 |

- Boxes past the right or bottom edge overshoot by a median of 22 px (max 2,080 px).
- **Handling:** `clip_boxes` trims each box to its tile and keeps it only if at least 50% of its *full* area is visible. Slight overshoots are kept; boxes mostly outside the image are dropped; zero-area boxes are dropped.

### Very large objects
- 117 boxes are larger than 1,000 px on a side, mostly Building (67) and Construction Site (37), which are areas rather than compact objects.
- **Known limitation:** a box larger than a 640 px tile can never be 50% inside one tile, so tiling drops all of them (0.02% of objects). Accepted for v1; revisit with larger tiles or a separate area-detection approach.

### Geography
- Images come from 56 distinct sites (DBSCAN, 50 km radius) spread across every inhabited continent.
- Site sizes are very uneven: median 10 images, mean 15, max 144 (one site holds 17% of all images).
- Neighbouring images share coverage and appearance, so a random tile-level split would leak between train and test.
- Holding out a whole continent is too costly: the Americas have 304 images (36%) and South America alone 245 images (29%, 8 sites).
- **Split plan:**
  - Hold out a subset of South American sites totalling about 10% of images, never used in training. Reserved as the "new region" for the drift and retraining demo. Exact sites are chosen in `split.py`.
  - Split the remaining sites into train/val/test by whole site, targeting about 70/15/15 of *images* (not sites), so the 144-image site can't dominate one split.

  - **Split (seed [0 or your best seed]), saved in `splits/image_splits.csv`:**

| Split | Images | Share |
| --- | --- | --- |
| train | 521 | 61.6% (69.9% of non-holdout) |
| val | 112 | 13.2% (15.0%) |
| test | 112 | 13.2% (15.0%) |
| holdout | 101 | 11.9%: 7 South American sites, never used in training |

- Objects per group and split:

| Group | train | val | test | holdout |
| --- | --- | --- | --- | --- |
| building | 157,220 | 57,357 | 64,895 | 41,361 |
| small_vehicle | 114,984 | 26,264 | 54,921 | 23,183 |
| large_vehicle | 22,319 | 8,908 | 8,675 | 3,166 |
| ship | 3,415 | 473 | 687 | 566 |
| storage_tank | 927 | 263 | 353 | 169 |
| aircraft | 659 | 218 | 186 | 168 |

- Every group appears in every split. Aircraft is the weakest: only 62% of non-holdout aircraft are in train, because airport sites are few and whole sites move together.

- Checked 20 seeds for a more even class mix; the best (seed 14) reduced the worst group's deviation from a 70% train share only from 0.141 to 0.133. Kept seed 0: the limit comes from keeping whole sites together (aircraft sit in a few airport sites), not from the split algorithm.

- Some objects are annotated twice with identical boxes; duplicates are dropped in `labels_table`.