"""Tests for the STRUCT4 / FULL13 label spaces and remap tables."""
import numpy as np

from src.label_spaces import (
    S3DIS_CLASSES,
    STRUCT4,
    to_struct4,
    map_string_to_struct4_idx,
)


def test_class_lists_lengths():
    assert len(S3DIS_CLASSES) == 13
    assert len(STRUCT4) == 4
    assert STRUCT4 == ["ceiling", "floor", "wall", "object"]


def test_to_struct4_structural_classes_map_to_themselves():
    labels13 = np.array([0, 1, 2])  # ceiling, floor, wall
    out = to_struct4(labels13)
    assert out.tolist() == [0, 1, 2]


def test_to_struct4_all_objects_collapse():
    # every non-structural class (3..12) must map to STRUCT4 'object' (index 3)
    labels13 = np.arange(3, 13)
    out = to_struct4(labels13)
    assert set(out.tolist()) == {3}


def test_to_struct4_full_range():
    out = to_struct4(np.arange(13))
    assert out.tolist() == [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]


def test_map_string_to_struct4_structural():
    assert map_string_to_struct4_idx("ceiling") == 0
    assert map_string_to_struct4_idx("floor") == 1
    assert map_string_to_struct4_idx("wall") == 2


def test_map_string_to_struct4_object_default():
    for s in ["object", "clutter", "unknown", "horizontal_surface", "chair", "anything"]:
        assert map_string_to_struct4_idx(s) == 3


def test_map_string_case_insensitive():
    assert map_string_to_struct4_idx("FLOOR") == 1
    assert map_string_to_struct4_idx("Wall") == 2


def test_struct4_remap_consistency_between_paths():
    # GT via to_struct4 and pred via string map must agree on structural names
    for i, name in enumerate(STRUCT4[:3]):
        assert map_string_to_struct4_idx(name) == i
