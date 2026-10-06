# Results

## Baseline: YOLO11s, 30 epochs (MLflow run `yolo11s-e30`)
- Data: tiles after decisions 001-011; 640 px; batch 12; seed 0. Trained on 11,6xx train tiles, evaluated on 3,863 val tiles.
- Training time: 1.8 h on an RTX A2000 8 GB laptop GPU. Inference: ~6 ms per tile.
- **Val mAP50 0.640, mAP50-95 0.322** (tile level; objects in overlapping strips are counted in both tiles).

| Group | Precision | Recall | AP50 | AP50-95 |
| --- | --- | --- | --- | --- |
| aircraft | 0.842 | 0.834 | 0.888 | 0.524 |
| small_vehicle | 0.696 | 0.708 | 0.699 | 0.254 |
| building | 0.661 | 0.579 | 0.637 | 0.321 |
| ship | 0.623 | 0.572 | 0.598 | 0.316 |
| storage_tank | 0.686 | 0.477 | 0.514 | 0.276 |
| large_vehicle | 0.643 | 0.444 | 0.507 | 0.242 |

### Observations
- Aircraft scores highest despite being the rarest group: large, distinctive shapes.
- Large vehicle is weakest (recall 0.44): the group mixes buses, trucks, rail and construction machines, and is often confused with small vehicles.
- Storage tank has low recall with reasonable precision, typical of few training examples.
- mAP50 plateaued around epoch 26 (0.60 at epoch 19, 0.64 at 26-30).
- GPU memory warnings during in-training validation were non-fatal; every epoch completed.

### Next experiments
- Image-level evaluation: merge detections across overlapping tiles, count each object once.
- Large-vehicle confusion: inspect the confusion matrix; consider splitting the group.
- Test split and the held-out South America region: only after choosing a final model.