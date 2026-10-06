"""Group the 60 xView classes into 6 coarse groups for v1 (see docs/decisions.md, 004)."""

GROUP_NAMES: tuple[str, ...] = (
    "aircraft",
    "small_vehicle",
    "large_vehicle",
    "ship",
    "building",
    "storage_tank",
)

GROUP_MEMBERS: dict[str, list[int]] = {
    "aircraft": [0, 1, 2, 3],
    "small_vehicle": [4, 5, 7, 8],
    "large_vehicle": [6, *range(9, 23), 33, *range(36, 46)],
    "ship": list(range(23, 33)),
    "building": list(range(46, 52)),
    "storage_tank": [55],
}

LEFT_OUT: frozenset[int] = frozenset({34, 35, 52, 53, 54, 56, 57, 58, 59})

CLASS_TO_GROUP: dict[int, int] = {
    c: g for g, name in enumerate(GROUP_NAMES) for c in GROUP_MEMBERS[name]
}


def to_group(class_index: int) -> int:
    """Map a 0-59 xView class index to a group index 0-5, or -1 if left out."""
    return CLASS_TO_GROUP.get(class_index, -1)
