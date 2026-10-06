# Data

## Source
xView 2018 (DIUx), satellite imagery at about 0.3 m per pixel, 60 object classes.
License: CC BY-NC-SA 4.0 (non-commercial use with attribution).
Not included in this repo; download manually after registering at challenge.xviewdataset.org.

## Files used
- train_images.zip: labelled GeoTIFFs
- train_labels.zip: xView_train.geojson (one feature per object)
- val_images.zip: unlabelled; used only for demo inference

Verify your download: `cd <download folder> && sha256sum -c docs/xview_checksums.sha256`

- Note: train_images.zip fails with Ubuntu's `unzip` ("possible zip bomb"); extract with `7z x train_images.zip`. Delete the `__MACOSX` folder it contains.
- 846 training images extracted (23 GB); 1395.tif is missing from the release.

## Layout
data/raw/train_images/*.tif
data/raw/xView_train.geojson

## What the data looks like

Findings from `notebooks/01_explore.ipynb`.

### Size
- 601,937 objects across 847 labelled images; 846 image files.
- `1395.tif` is missing from the release, so its 131 objects are skipped.
- 79 objects have a `type_id` that is not one of the 60 classes; dropped.

### Class imbalance
- Building (316,795; 53%) and Small Car (211,664; 35%) make up 88% of all objects.
- The third-largest class, Truck, has 12,189 (2%). Many classes have only a few hundred examples.
- **Implication:** report mAP per class, not only overall; a model that only finds buildings and cars would still score well overall.

### Object size
- Median box is 25 x 23 px; 5% are 9 px or smaller; 0.9% are under 10 px on their longest side.
- Images are ~3,300 x 2,900 px, so objects are tiny relative to the image.
- **Implication:** tile at full resolution instead of downscaling; shrinking a whole image would turn a 25 px car into ~5 px.

### Image format
- 3-band RGB GeoTIFFs, all in EPSG:4326, ~3 x 10^-6 degrees per pixel (~0.3 m).

### Label quality
9,937 boxes (1.65%) fail at least one basic check (a box can fail more than one):

| Problem | Boxes |
| --- | --- |
| Starts before the image edge (negative coordinates) | 5,058 |
| Extends past the right edge | 2,615 |
| Extends past the bottom edge | 2,368 |
| Zero or negative width/height | 9 |

- Boxes past the right or bottom edge overshoot by a median of 22 px (max 2,080 px).
- **Handling:** `clip_boxes` trims each box to its tile, and keeps it only if at least 50% of its *full* area is visible. Slight overshoots are kept; boxes mostly outside the image are dropped; zero-area boxes are dropped.

### Very large objects
- 117 boxes are larger than 1,000 px on a side, mostly Building (67) and Construction Site (37), which are areas rather than compact objects.
- **Known limitation:** a box larger than a 640 px tile can never be 50% inside one tile, so tiling drops all of them (0.02% of objects). Accepted for v1; revisit with larger tiles or a separate area-detection approach.

### Geography
- Images come from [N] distinct geographic clusters (Cell 7). Train/test will be split by cluster, not at random, to avoid leakage between neighbouring tiles.