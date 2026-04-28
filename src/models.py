"""
src/models.py — Data Blueprints (Pydantic models)

This file defines what data looks like throughout the whole pipeline. It uses Pydantic 
— a Python library that validates data automatically.

There are two main result types:
1. PlaneResult — describes one detected flat surface (floor, wall, ceiling):
   - plane_id — which plane number is this
   - label — "floor", "wall", "ceiling", "unknown"
   - normal — a 3D vector pointing perpendicular to the surface (e.g. the floor's normal points straight up: [0, 0, 1])
   - centroid_z — average height of this plane
   - inlier_cloud — the actual 3D points that belong to this plane

2. ClusterResult — describes one detected object (chair, table, etc.):
   - cluster_id, label, confidence — what object, how sure are we
   - dims — [width, depth, height] in metres
   - centroid, z_min, z_max — where is it in 3D space
   - footprint_m2 — how much floor space it takes up

3. PipelineConfig — wraps all the settings from your YAML file with validation. 
   If you type a wrong value in the YAML (like voxel_size: -5), Pydantic will immediately tell you it's wrong.

Why this matters in interviews: Pydantic gives you free data validation, type safety, 
and JSON serialization. The interviewer will appreciate knowing you understand how to 
enforce data integrity in a complex pipeline.
"""

import open3d as o3d                      # Import Open3D for 3D data structures (PointClouds, BoundingBoxes)
from typing import List, Optional, Dict   # Import typing utilities for better code documentation and static analysis
from pydantic import BaseModel, Field, field_validator, model_validator # Import Pydantic for data validation and schema definition

# ── Configuration Models ─────────────────────────────────────────────────────

# INTERVIEW TIP: Why Pydantic? 
# Interviewers might ask why we use Pydantic models. 
# Answer: It provides runtime type validation, clear schema definition, 
# and easy serialization/deserialization for the configuration and reports.

class SORConfig(BaseModel):
    """Configuration for Statistical Outlier Removal."""
    nb_neighbors: int = Field(20, gt=0) # Number of neighbors to analyze for each point (must be > 0)
    std_ratio: float = Field(2.0, gt=0) # Standard deviation threshold; lower means more aggressive cleaning

class NormalEstimationConfig(BaseModel):
    """
    Settings for computing surface normals.
    Normals are required for RANSAC and Region Growing to understand surface orientation.
    """
    radius: float = Field(0.1, gt=0)   # The search radius (in meters) for finding neighboring points
    max_nn: int = Field(30, gt=0)     # The maximum number of neighbors to use for normal calculation
    orient_k: int = Field(15, gt=0)   # Number of neighbors used to ensure all normals point in a consistent direction

class RORConfig(BaseModel):
    """Configuration for Radius Outlier Removal."""
    radius: float = Field(0.1, gt=0)         # Search radius to check for neighboring points
    min_neighbors: int = Field(10, gt=0)     # Minimum points required within the radius to not be considered noise

class PreprocessingConfig(BaseModel):
    """
    Groups all preprocessing hyperparameters. 
    Voxel size controls the resolution of the downsampled cloud.
    """
    voxel_size: float = Field(0.05, gt=0)                     # Size of the cubes used for downsampling (5cm)
    sor: SORConfig = Field(default_factory=SORConfig)         # Nested configuration for statistical cleaning
    ror: RORConfig = Field(default_factory=RORConfig)         # Nested configuration for radius-based cleaning
    normal_estimation: NormalEstimationConfig = Field(default_factory=NormalEstimationConfig) # Settings for normals

class RansacConfig(BaseModel):
    """
    RANSAC (Random Sample Consensus) Hyperparameters for plane detection.
    """
    distance_threshold: float = Field(0.02, gt=0) # Max distance (2cm) from a plane for a point to be an "inlier"
    ransac_n: int = Field(3, ge=3)               # Minimum points needed to define a plane model (mathematically 3)
    num_iterations: int = Field(2000, gt=0)      # Number of random samples to try; more means better accuracy
    min_plane_size: int = Field(1000, gt=0)      # Minimum number of points required to accept a detected plane
    max_planes: int = Field(10, gt=0)            # Limit on how many different planes we try to find in one scene
    remaining_points_min: int = Field(500, ge=0) # Stop looking for planes if the point cloud gets too small

class DbscanConfig(BaseModel):
    """Configuration for DBSCAN clustering (Density-Based Spatial Clustering)."""
    eps: float = Field(0.1, gt=0)                # Max distance (10cm) between two points to be in the same cluster
    min_points: int = Field(50, gt=0)            # Minimum points to form a dense "core" of a cluster
    min_cluster_points: int = Field(100, gt=0)   # Minimum total points for a cluster to be considered an object
    max_object_size: float = Field(3.0, gt=0)    # Ignore clusters larger than this (e.g., 3 meters) as they are likely noise

class LabelingConfig(BaseModel):
    """
    Heuristics for semantic classification (turning clusters into 'chairs', 'tables', etc.).
    """
    floor_z_threshold: float = Field(0.15)            # Any plane below 15cm is likely the floor
    ceiling_z_fraction: float = Field(0.8, ge=0, le=1.0) # Planes in the top 20% of the room height are ceilings
    horizontal_angle_deg: float = Field(15, ge=0, le=90) # Max tilt angle for a surface to be considered "flat/horizontal"
    vertical_angle_deg: float = Field(75, ge=0, le=90)   # Min angle for a surface to be considered a "vertical wall"
    tall_furniture_min_h: float = Field(1.5, gt=0)      # Objects taller than 1.5m are labeled as tall furniture
    furniture_min_footprint: float = Field(0.3, gt=0)   # Objects must cover at least 0.3m^2 to be furniture
    furniture_min_h: float = Field(0.3, gt=0)           # Objects must be at least 30cm tall to be furniture
    small_object_max_footprint: float = Field(0.5, gt=0)# Small objects (like boxes) must be smaller than this area
    high_fixture_min_z: float = Field(1.5, gt=0)        # Objects that don't touch the floor and start high up (e.g., lamps)

class OutputConfig(BaseModel):
    """Paths for where to save the results of the pipeline."""
    ply: str = "outputs/segmented_room.ply"            # Path for the colored 3D point cloud file
    report: str = "outputs/segmentation_report.json"    # Path for the data-rich JSON summary
    screenshot: str = "outputs/segmentation_viz.png"   # Path for the static visualization image

class PipelineConfig(BaseModel):
    """Top-level configuration model that combines all sub-configs."""
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig) # Cleaning settings
    ransac: RansacConfig = Field(default_factory=RansacConfig)                      # Plane detection settings
    dbscan: DbscanConfig = Field(default_factory=DbscanConfig)                      # Object clustering settings
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)                # Classification settings
    output: OutputConfig = Field(default_factory=OutputConfig)                      # File saving settings

# ── Plane result ─────────────────────────────────────────────────────────────

class PlaneResult(BaseModel):
    """Represents a geometric plane (floor, wall) detected via RANSAC."""
    model_config = {"arbitrary_types_allowed": True} # Allows including Open3D objects in the model

    def __getitem__(self, item):
        """Enable dictionary-like access (e.g., plane['label']) for testing compatibility."""
        return getattr(self, item)

    def __contains__(self, item):
        """Check if a field exists in the model."""
        return hasattr(self, item)

    def keys(self):
        """Return all field names in the model."""
        return self.model_fields.keys()

    plane_id: int = Field(..., ge=0, description="Sequential index of the plane.") # Unique number for the plane
    label: str = Field("unknown", description="Semantic tag (floor, wall, etc.).") # Human-readable label
    inlier_count: int = Field(..., gt=0)               # How many 3D points actually lie on this plane
    centroid_z: float                                  # The average height (vertical position) of the plane
    normal: List[float] = Field(..., min_length=3, max_length=3) # [nx, ny, nz] direction vector
    plane_model: List[float] = Field(..., min_length=4, max_length=4,
                                     description="[a, b, c, d] from ax + by + cz + d = 0") # Math equation of the plane
    
    # This stores the actual points but won't be included in the JSON export
    inlier_cloud: Optional[o3d.geometry.PointCloud] = Field(None, exclude=True)

    @field_validator("normal")
    @classmethod
    def normal_not_all_zero(cls, v: List[float]) -> List[float]:
        """Validator to ensure the normal vector isn't mathematically impossible (0,0,0)."""
        if all(abs(x) < 1e-9 for x in v):
            raise ValueError("Normal vector must not be the zero vector.")
        return v

    @field_validator("label")
    @classmethod
    def label_is_known(cls, v: str) -> str:
        """Validator to catch typos in labels and ensure they match our system's logic."""
        valid = {
            "floor", "ceiling", "wall", "horizontal_surface",
            "unknown", "noise"
        }
        if v not in valid:
            raise ValueError(f"Plane label '{v}' is invalid. Valid: {sorted(valid)}")
        return v

# ── Cluster result ────────────────────────────────────────────────────────────

class ClusterResult(BaseModel):
    """Represents an object cluster (chair, table) detected via DBSCAN."""
    model_config = {"arbitrary_types_allowed": True} # Allows including Open3D geometry objects

    def __getitem__(self, item):
        """Enable dictionary-like access; handles special case for bounding boxes."""
        if item == 'aabb':
            return getattr(self, 'aabb_box', None)
        return getattr(self, item)

    def __contains__(self, item):
        """Check if a field exists."""
        return hasattr(self, item)

    def keys(self):
        """Return all field names."""
        return self.model_fields.keys()

    cluster_id: int = Field(..., ge=0)                 # Unique identifier for the object
    label: str = Field("unknown")                      # Semantic label (e.g., 'chair')
    confidence: float = Field(0.5, ge=0.0, le=1.0)     # Probability score of the label being correct
    n_points: int = Field(..., gt=0)                   # Number of points that make up the object
    point_density: float = Field(0.0, ge=0.0)          # Points per m^3 (useful for distinguishing materials)
    centroid: List[float] = Field(..., min_length=3, max_length=3) # [x, y, z] center point
    dims: List[float] = Field(..., min_length=3, max_length=3,
                               description="[width, depth, height] in meters.") # Physical size
    z_min: float                                       # Minimum height (contact point with floor)
    z_max: float                                       # Maximum height (top of the object)
    footprint_m2: float = Field(..., ge=0.0)           # Projected floor area (width * depth)

    # These fields are filled later by the BoundingBoxEstimator module
    obb_extent: Optional[List[float]] = None           # Dimensions of the oriented bounding box
    obb_rotation_deg: Optional[float] = None           # Rotation angle of the object in the room
    
    # These geometry objects are excluded from the JSON report to save space
    cloud: Optional[o3d.geometry.PointCloud] = Field(None, exclude=True)
    aabb_box: Optional[o3d.geometry.AxisAlignedBoundingBox] = Field(None, exclude=True)
    obb_box: Optional[o3d.geometry.OrientedBoundingBox] = Field(None, exclude=True)
# Pydantic gives you free data validation, type safety, and JSON serialization
    @model_validator(mode="after")
    def z_min_le_z_max(self) -> "ClusterResult":
        """Validator to ensure the bottom of an object isn't higher than its top."""
        if self.z_min > self.z_max + 1e-6:
            raise ValueError(f"z_min ({self.z_min}) must be <= z_max ({self.z_max}).")
        return self

    @field_validator("dims")
    @classmethod
    def dims_non_negative(cls, v: List[float]) -> List[float]:
        """Validator to ensure physical dimensions are never negative."""
        if any(d < -1e-6 for d in v):
            raise ValueError(f"Dimensions must be non-negative, got {v}.")
        return v

    @field_validator("label")
    @classmethod
    def label_is_known(cls, v: str) -> str:
        """Validator to ensure semantic labels match the predefined vocabulary."""
        valid = {
            "furniture", "tall_furniture", "small_object",
            "high_fixture", "horizontal_surface", "unknown", "noise",
            "chair", "table", "shelf"
        }
        if v not in valid:
            raise ValueError(f"Cluster label '{v}' is invalid. Valid: {sorted(valid)}")
        return v

# ── Segmentation report ───────────────────────────────────────────────────────

class SegmentationReport(BaseModel):
    """Top-level JSON structure that holds all results for the entire scene."""
    structural_planes: List[PlaneResult] = Field(default_factory=list) # List of all detected walls/floors
    objects: List[ClusterResult] = Field(default_factory=list)         # List of all detected furniture/objects

    @property
    def plane_label_counts(self) -> dict:
        """Helper to quickly see how many walls, floors, etc. were found."""
        counts: dict = {}
        for p in self.structural_planes:
            counts[p.label] = counts.get(p.label, 0) + 1
        return counts

    @property
    def object_label_counts(self) -> dict:
        """Helper to quickly see how many chairs, tables, etc. were found."""
        counts: dict = {}
        for c in self.objects:
            counts[c.label] = counts.get(c.label, 0) + 1
        return counts
