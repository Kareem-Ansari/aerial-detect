# Design decisions

Each entry records the context, the decision, why, and what was traded off.

## 001: Tile size 640 px, 20% overlap, last tile snapped to the edge
- Context: xView images are about 3,300 x 2,900 px; detectors train on fixed-size inputs; objects are tiny (median 25 px).
- Decision: 640 x 640 tiles with 20% overlap (stride 512). The last tile in each row and column is moved back so it ends exactly at the image edge.
- Why: 640 px is a standard detector input and fits comfortably in batches on an 8 GB GPU. With overlap, objects up to about 128 px cut by one tile's edge still appear whole in a neighbouring tile. Snapping keeps every tile full size with real pixels.
- Trade-off: about 1.5x more tiles to store and process; duplicate detections must be merged at inference. Edge tiles overlap their neighbours by more than 20%.

## 002: Images smaller than one tile
- Decision: return a single 640 x 640 window and zero-pad the image when reading it.
- Why: keeps the code general and every tile the same size. In practice no xView image is smaller than 640 px (minimum 2,576 x 2,426), so this rarely runs.
- Trade-off: padded pixels carry no information; acceptable since it almost never happens.

## 003: Keep a box only if at least 50% of it is inside the tile
- Decision: `clip_boxes` keeps a box when its visible area is at least 50% of its full area, and returns the indices of kept boxes so class labels stay aligned.
- Why: slivers of an object don't look like the object and would teach the model wrong shapes; thanks to overlap, the same object usually appears whole in a neighbouring tile. It also handles the 1.65% of labels that extend past the image edge.
- Trade-off: boxes larger than a tile (117 objects, 0.02%) can never be kept. Measured and accepted for v1.

## 004: Group 60 classes into 6 for v1
- Context: Building (53%) and Small Car (35%) dominate; most of the other 58 classes have a few thousand examples or fewer.
- Decision: Aircraft (0-3), Small vehicle (4, 5, 7, 8), Large vehicle (6, 9-22, 33, 36-45), Ship (23-32), Building (46-51), Storage tank (55).
  Left out: cranes, construction sites, lots, containers, pylons, towers (34, 35, 52-54, 56-59).
- Why: rare classes can't be learned reliably yet. Several left-out classes are areas (Construction Site, Vehicle Lot), not compact objects; 37 of the 117 oversized boxes are Construction Sites.
- Trade-off: lose fine-grained labels such as truck types; can be added back in v2.

## 005: Split by geographic site; hold out part of South America
- Context: images cluster into 56 sites (median 10 images, max 144). Neighbouring images share coverage.
- Considered: holding out all of the Americas (304 images, 36%) or all of South America (245 images, 29%). Rejected; both leave too little training data.
- Decision: hold out a subset of South American sites totalling about 10% of images as a never-seen region. Split the remaining sites about 70/15/15 by image count, keeping each site whole.
- Why: a random split leaks near-duplicate tiles into test and inflates scores, the same failure mode as my thesis's camera-position leak. The held-out region gives a real drift scenario to monitor and retrain on.
- Trade-off: test scores will be lower but honest. Planned check: train once with a random split and once by site, and report the gap.

## 006: Store tiles as JPEG (quality 95)
- Context: about 35,000 tiles of 640 x 640 RGB.
- Decision: JPEG at quality 95 instead of PNG.
- Why: about 5-8 GB instead of about 30 GB; compression artifacts at q95 are far below what affects detection.
- Trade-off: lossy; if small-object accuracy turns out sensitive to it, compare against PNG on a sample.

## 007: Keep 10% of empty tiles in train, all in val/test/holdout
- Decision: tiles with no objects are kept with probability 0.10 in train (seeded per tile id), and always kept elsewhere.
- Why: thousands of empty tiles slow training without teaching much, but evaluation must include empty ground to measure false positives honestly. Seeding by tile id makes the choice identical across runs even with parallel workers.
- Trade-off: the model sees less empty background in training; watch the false-positive rate on val.