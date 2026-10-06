from aerial_detect.classes import NAMES, XVIEW_CLASS2INDEX, to_index


def test_sixty_classes() -> None:
    assert len(NAMES) == 60


def test_mapping_covers_every_class_exactly_once() -> None:
    valid = sorted(i for i in XVIEW_CLASS2INDEX if i >= 0)
    assert valid == list(range(60))


def test_known_ids() -> None:
    assert to_index(11) == 0  # Fixed-wing Aircraft
    assert to_index(94) == 59  # Tower


def test_unused_and_out_of_range_ids() -> None:
    assert to_index(14) == -1
    assert to_index(0) == -1
    assert to_index(500) == -1
    assert to_index(-3) == -1
