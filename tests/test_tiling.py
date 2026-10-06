import numpy as np
import pytest

from aerial_detect.tiling import clip_boxes, tile_windows

# ---------- tile_windows ----------


def test_single_tile_when_image_equals_tile() -> None:
    assert tile_windows(640, 640) == [(0, 0, 640, 640)]


def test_last_tile_snaps_to_edge() -> None:
    windows = tile_windows(1000, 1000, tile=640, overlap=0.2)
    starts = sorted({(x0, y0) for x0, y0, _, _ in windows})
    assert starts == [(0, 0), (0, 360), (360, 0), (360, 360)]


def test_all_tiles_are_full_size() -> None:
    for x0, y0, x1, y1 in tile_windows(2000, 1500):
        assert (x1 - x0, y1 - y0) == (640, 640)


def test_every_pixel_is_covered() -> None:
    w, h = 2000, 1500
    covered = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in tile_windows(w, h):
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_image_smaller_than_tile_gives_one_window() -> None:
    assert tile_windows(300, 200) == [(0, 0, 640, 640)]


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])
def test_invalid_overlap_raises(overlap: float) -> None:
    with pytest.raises(ValueError):
        tile_windows(1000, 1000, overlap=overlap)


# ---------- clip_boxes ----------


def test_box_inside_is_shifted_not_resized() -> None:
    boxes = np.array([[400, 100, 440, 140]], dtype=float)
    out, keep = clip_boxes(boxes, (360, 0, 1000, 640))
    np.testing.assert_allclose(out, [[40, 100, 80, 140]])
    assert keep.tolist() == [0]


def test_box_mostly_inside_is_clipped_and_kept() -> None:
    boxes = np.array([[576, 100, 676, 200]], dtype=float)  # 64% inside
    out, keep = clip_boxes(boxes, (0, 0, 640, 640), min_visible=0.5)
    np.testing.assert_allclose(out, [[576, 100, 640, 200]])
    assert keep.tolist() == [0]


def test_box_mostly_outside_is_dropped() -> None:
    boxes = np.array([[630, 100, 730, 200]], dtype=float)  # 10% inside
    out, keep = clip_boxes(boxes, (0, 0, 640, 640), min_visible=0.5)
    assert out.shape == (0, 4)
    assert keep.tolist() == []


def test_keep_indices_track_original_rows() -> None:
    boxes = np.array([[10, 10, 50, 50], [900, 900, 950, 950], [100, 100, 150, 150]], dtype=float)
    _, keep = clip_boxes(boxes, (0, 0, 640, 640))
    assert keep.tolist() == [0, 2]


def test_zero_area_box_is_dropped() -> None:
    boxes = np.array([[10, 10, 10, 50]], dtype=float)
    out, _ = clip_boxes(boxes, (0, 0, 640, 640))
    assert out.shape == (0, 4)


def test_empty_input_returns_empty() -> None:
    out, keep = clip_boxes(np.zeros((0, 4)), (0, 0, 640, 640))
    assert out.shape == (0, 4)
    assert keep.shape == (0,)
