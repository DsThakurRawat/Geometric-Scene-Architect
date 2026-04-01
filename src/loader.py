import open3d as o3d
import numpy as np
from pathlib import Path
from typing import Dict, Any


class PointCloudLoader:
    """
    Module 1: Data Loading & Validation
    Handles reading point clouds, providing basic statistics, and normalizing orientation.
    """

    SUPPORTED_EXTENSIONS = [".ply", ".pcd", ".xyz", ".xyzrgb", ".pts"]

    def load(self, path: str) -> o3d.geometry.PointCloud:
        """Loads a point cloud from the specified path."""
        file_path = Path(path)
        ext = file_path.suffix.lower()

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        pcd = o3d.io.read_point_cloud(str(file_path))

        if len(pcd.points) == 0:
            raise ValueError(f"Point cloud at '{path}' is empty (0 points loaded).")

        return pcd

    def validate(self, pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """Returns a dictionary with basic statistics about the point cloud."""
        pts = np.asarray(pcd.points)

        if len(pts) == 0:
            return {
                "n_points": 0,
                "has_colors": False,
                "has_normals": False,
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [0.0, 0.0, 0.0],
                "scene_dims": [0.0, 0.0, 0.0],
                "centroid": [0.0, 0.0, 0.0],
            }

        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        dims = bbox_max - bbox_min

        return {
            "n_points": len(pts),
            "has_colors": pcd.has_colors(),
            "has_normals": pcd.has_normals(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "scene_dims": dims.tolist(),
            "centroid": pts.mean(axis=0).tolist(),
        }

    def normalize_orientation(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Translates the cloud so the floor (5th-percentile Z) sits at Z=0.
        Operates in-place and returns the same object.
        """
        pts = np.asarray(pcd.points)
        if len(pts) == 0:
            return pcd

        z_floor = float(np.percentile(pts[:, 2], 5))
        pcd.translate([0.0, 0.0, -z_floor])
        return pcd
