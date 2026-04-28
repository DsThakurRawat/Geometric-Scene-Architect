"""
src/loader.py — Module 1: Load & Normalize

What it does: Reads your 3D point cloud file from disk.
Supported formats: .ply, .pcd, .xyz, .xyzrgb, .pts

The key method is normalize_orientation() — it finds the 5th percentile of Z values 
(which is roughly the floor) and shifts all points down so the floor sits at Z=0. 
This is important because a scanned room might have the floor at Z=2.3 metres for 
no particular reason.

Example code:
z_floor = np.percentile(pts[:, 2], 5)  # find floor height
pcd.translate([0, 0, -z_floor])         # shift everything down

Why 5th percentile instead of the minimum? Because the minimum could be noise or 
a stray point below the floor. The 5th percentile is more robust.
"""

import open3d as o3d                      # Import Open3D for 3D data handling and processing
import numpy as np                        # Import NumPy for numerical operations and array manipulation
from pathlib import Path                  # Import Path for cross-platform file path management
from typing import Dict, Any              # Import typing hints for dictionary and general types

# Module 1: Load & Normalize
# Open3D = a Python library for 3D data (point clouds, meshes, 3D visualization, geometry)
"""
Used in:
- 3D vision
- LiDAR / point clouds
- robotics / AR
"""

class PointCloudLoader:
    """
    Module 1: Data Loading & Validation
    Handles reading point clouds, providing basic statistics, and normalizing orientation.
    """

    # List of file extensions that the pipeline can process
    SUPPORTED_EXTENSIONS = [".ply", ".pcd", ".xyz", ".xyzrgb", ".pts"]

    def load(self, path: str) -> o3d.geometry.PointCloud:
        """Loads a point cloud from the specified path."""
        file_path = Path(path)            # Create a Path object for the input string
        ext = file_path.suffix.lower()    # Get the file extension and convert to lowercase

        # Check if the file format is supported by the pipeline
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: '{ext}'. "
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )                             # Raise error if format is unknown

        # Verify that the file actually exists on the system
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}") # Raise error if path is invalid

        # Use Open3D to read the point cloud file into memory
        pcd = o3d.io.read_point_cloud(str(file_path))

        # Check if the loaded cloud actually contains data points
        if len(pcd.points) == 0:
            raise ValueError(f"Point cloud at '{path}' is empty (0 points loaded).")

        # Return the resulting Open3D PointCloud object
        return pcd

    def validate(self, pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
        """Returns a dictionary with basic statistics about the point cloud."""
        pts = np.asarray(pcd.points)      # Convert the points to a NumPy array for math

        # If the cloud has no points, return a dictionary of default/zero values
        if len(pts) == 0:
            return {
                "n_points": 0,            # Zero points found
                "has_colors": False,      # No color data
                "has_normals": False,     # No normal vectors
                "bbox_min": [0.0, 0.0, 0.0], # Default bounding box min
                "bbox_max": [0.0, 0.0, 0.0], # Default bounding box max
                "scene_dims": [0.0, 0.0, 0.0], # Default scene size
                "centroid": [0.0, 0.0, 0.0],   # Default center point
            }

        # Find the smallest and largest coordinates in the scene
        bbox_min = pts.min(axis=0)        # Minimum X, Y, Z coordinates
        bbox_max = pts.max(axis=0)        # Maximum X, Y, Z coordinates
        dims = bbox_max - bbox_min        # Subtract min from max to get total scene dimensions

        # Compile and return the gathered statistics as a dictionary
        return {
            "n_points": len(pts),         # Total number of points
            "has_colors": pcd.has_colors(), # Whether color information exists
            "has_normals": pcd.has_normals(), # Whether surface normals exist
            "bbox_min": bbox_min.tolist(),   # List of [min_x, min_y, min_z]
            "bbox_max": bbox_max.tolist(),   # List of [max_x, max_y, max_z]
            "scene_dims": dims.tolist(),     # List of [width, depth, height]
            "centroid": pts.mean(axis=0).tolist(), # Average position [avg_x, avg_y, avg_z]
        }

    def normalize_orientation(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Translates the cloud so the floor (5th-percentile Z) sits at Z=0.
        Operates in-place and returns the same object.
        
        INTERVIEW TIP: Why 5th percentile?
        If we use the MIN (0th percentile), a single outlier point below the floor 
        would shift the entire scene. Using the 5th percentile is more robust to noise.
        """
        pts = np.asarray(pcd.points)      # Get the point data as a NumPy array
        
        # If there are no points, we can't normalize anything
        if len(pts) == 0:
            return pcd

        # Calculate the 5th percentile of the Z-axis to find the floor level
        z_floor = float(np.percentile(pts[:, 2], 5))
        
        # Shift the entire point cloud so that the calculated floor is at Z=0
        pcd.translate([0.0, 0.0, -z_floor])
        
        # Return the modified point cloud object
        return pcd
