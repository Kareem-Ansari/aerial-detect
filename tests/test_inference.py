import numpy as np

from aerial_detect.inference import TileDetections, merge_detections, nms, predict_image


def test_nms_keeps_highest_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=float)
    assert nms(boxes, np.array([0.6, 0.9])).tolist() == [1]


def test_nms_keeps_separate_boxes() -> None:
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=float)
    assert sorted(nms(boxes, np.array([0.5, 0.9])).tolist()) == [0, 1]


def test_nms_drops_partial_box_inside_full_box() -> None:
    # IoU is only 0.4, but the partial box lies entirely inside the full one
    boxes = np.array([[0, 0, 100, 100], [60, 0, 100, 100]], dtype=float)
    assert nms(boxes, np.array([0.9, 0.8])).tolist() == [0]


def test_merge_keeps_overlapping_boxes_of_different_classes() -> None:
    boxes = np.array([[0, 0, 100, 100], [0, 0, 100, 100]], dtype=float)
    _, _, classes = merge_detections(boxes, np.array([0.9, 0.8]), np.array([1, 4]))
    assert sorted(classes.tolist()) == [1, 4]


def test_predict_image_shifts_boxes_and_merges_across_tiles() -> None:
    # 1000 x 640 image -> tiles at x = 0 and x = 360; the same object is seen in both
    def predictor(tiles: list[np.ndarray]) -> list[TileDetections]:
        assert len(tiles) == 2
        return [
            (np.array([[500.0, 100, 560, 160]]), np.array([0.9]), np.array([4])),
            (np.array([[140.0, 100, 200, 160]]), np.array([0.8]), np.array([4])),  # 140 + 360
        ]

    boxes, scores, _ = predict_image(np.zeros((640, 1000, 3), dtype=np.uint8), predictor)
    assert boxes.tolist() == [[500.0, 100.0, 560.0, 160.0]]
    assert scores.tolist() == [0.9]


def test_predict_image_with_no_detections() -> None:
    def predictor(tiles: list[np.ndarray]) -> list[TileDetections]:
        return [(np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)) for _ in tiles]

    boxes, scores, classes = predict_image(np.zeros((700, 700, 3), dtype=np.uint8), predictor)
    assert boxes.shape == (0, 4) and len(scores) == 0 and len(classes) == 0
