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
        # Create a Path object for easier manipulation of the file extension.
        file_path = Path(path)
        # Extract the suffix (e.g., '.ply') and convert to lowercase for comparison.
        ext = file_path.suffix.lower()

        # Check if the file format is supported by Open3D.
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        # Verify that the file actually exists on the filesystem.
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Use Open3D's IO module to read the point cloud.
        pcd = o3d.io.read_point_cloud(str(file_path))

        # Check if the cloud contains any points.
        if len(pcd.points) == 0:
            raise ValueError(f"Point cloud at '{path}' is empty (0 points loaded).")

        # Return the loaded PointCloud object.
        return pcd

    def validate(self, pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """Returns a dictionary with basic statistics about the point cloud."""
        # Convert Open3D points to a NumPy array for math operations.
        pts = np.asarray(pcd.points)

        # If the cloud is empty, return a dictionary of zero/default values.
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

        # Calculate the minimum and maximum coordinates in X, Y, and Z.
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        # Calculate the scene dimensions (width, depth, height).
        dims = bbox_max - bbox_min

        # Compile and return the statistics.
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
        
        INTERVIEW TIP: Why 5th percentile?
        If we use the MIN (0th percentile), a single outlier point below the floor 
        would shift the entire scene. Using the 5th percentile is more robust to noise.
        """
        # Convert points to NumPy array.
        pts = np.asarray(pcd.points)
        # Guard clause for empty clouds.
        if len(pts) == 0:
            return pcd

        # Calculate the height of the floor using the 5th percentile.
        z_floor = float(np.percentile(pts[:, 2], 5))
        # Move the entire cloud vertically so the floor is at Z=0.
        pcd.translate([0.0, 0.0, -z_floor])
        # Return the adjusted cloud.
        return pcd
