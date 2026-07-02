# src/label_spaces.py
import numpy as np

S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter"
]

STRUCT4 = ["ceiling", "floor", "wall", "object"]

# Mapping from FULL13 class indices to STRUCT4 class indices
# 0 (ceiling) -> 0
# 1 (floor) -> 1
# 2 (wall) -> 2
# 3 to 12 -> 3 (object)
_MAP_13_TO_4 = np.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.int32)

def to_struct4(labels13: np.ndarray) -> np.ndarray:
    """
    Remaps labels from the full 13-class space (FULL13) to the 4-class structural space (STRUCT4).
    """
    return _MAP_13_TO_4[labels13]

def map_string_to_struct4_idx(label_str: str) -> int:
    """
    Maps a string label from the geometry pipeline to the corresponding index in STRUCT4.
    """
    label_str = label_str.lower()
    if label_str == "ceiling":
        return 0
    elif label_str == "floor":
        return 1
    elif label_str == "wall":
        return 2
    else:
        return 3  # All other object types map to "object"
