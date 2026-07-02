# src/s3dis_loader.py
import os
import numpy as np
from typing import Tuple, Generator

def load_room(room_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads preprocessed S3DIS room points and labels from the cached directory.
    
    Args:
        room_dir: Path to the cached room directory containing points.npy and labels.npy
        
    Returns:
        points: np.ndarray of shape (N, 6) containing XYZRGB values
        gt_labels: np.ndarray of shape (N,) containing 13-class labels
    """
    points_path = os.path.join(room_dir, "points.npy")
    labels_path = os.path.join(room_dir, "labels.npy")
    
    if not os.path.exists(points_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"Preprocessed cache files not found in {room_dir}. "
            "Please run prepare_s3dis.py first."
        )
        
    points = np.load(points_path)
    gt_labels = np.load(labels_path)
    return points, gt_labels

def iter_area(area_id: int, processed_dir: str = "data/s3dis/processed") -> Generator[Tuple[str, np.ndarray, np.ndarray], None, None]:
    """
    Generator that yields preprocessed room data for a given Area.
    
    Args:
        area_id: Area number (e.g., 5)
        processed_dir: Path to the base processed cache directory
        
    Yields:
        room_name: Name of the room
        points: np.ndarray of shape (N, 6)
        gt_labels: np.ndarray of shape (N,)
    """
    area_name = f"Area_{area_id}"
    area_path = os.path.join(processed_dir, area_name)
    
    if not os.path.exists(area_path):
        raise FileNotFoundError(f"Area directory '{area_path}' does not exist.")
        
    for room_name in sorted(os.listdir(area_path)):
        room_dir = os.path.join(area_path, room_name)
        if os.path.isdir(room_dir):
            points, labels = load_room(room_dir)
            yield room_name, points, labels
