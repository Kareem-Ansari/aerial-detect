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
(filled in from notebooks/01_explore.ipynb)