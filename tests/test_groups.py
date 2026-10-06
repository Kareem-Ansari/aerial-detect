from aerial_detect.groups import CLASS_TO_GROUP, GROUP_MEMBERS, GROUP_NAMES, LEFT_OUT, to_group


def test_six_groups() -> None:
    assert len(GROUP_NAMES) == 6


def test_no_class_in_two_groups() -> None:
    members = [c for cs in GROUP_MEMBERS.values() for c in cs]
    assert len(members) == len(set(members))


def test_every_class_is_grouped_or_left_out() -> None:
    assert set(CLASS_TO_GROUP) | LEFT_OUT == set(range(60))
    assert not set(CLASS_TO_GROUP) & LEFT_OUT


def test_examples() -> None:
    assert GROUP_NAMES[to_group(48)] == "building"
    assert GROUP_NAMES[to_group(5)] == "small_vehicle"
    assert to_group(34) == -1  # tower crane, left out
    assert to_group(-1) == -1  # not a class
