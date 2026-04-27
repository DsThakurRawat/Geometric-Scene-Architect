import open3d as o3d
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Configuration Models ─────────────────────────────────────────────────────

# INTERVIEW TIP: Why Pydantic? 
# Interviewers might ask why we use Pydantic models. 
# Answer: It provides runtime type validation, clear schema definition, 
# and easy serialization/deserialization for the configuration and reports.

class SORConfig(BaseModel):
    nb_neighbors: int = Field(20, gt=0) # Number of neighbors to analyze for outlier removal
    std_ratio: float = Field(2.0, gt=0) # Standard deviation multiplier for thresholding

class NormalEstimationConfig(BaseModel):
    """
    Settings for computing surface normals.
    Normals are required for RANSAC and Region Growing to understand surface orientation.
    """
    radius: float = Field(0.1, gt=0)   # Radius of neighborhood to consider for each point
    max_nn: int = Field(30, gt=0)     # Maximum neighbors to search within the radius
    orient_k: int = Field(15, gt=0)   # Neighbors used to orient normals consistently

class RORConfig(BaseModel):
    radius: float = Field(0.1, gt=0)
    min_neighbors: int = Field(10, gt=0)

class PreprocessingConfig(BaseModel):
    """
    Groups all preprocessing hyperparameters. 
    Voxel size controls the resolution of the downsampled cloud.
    """
    voxel_size: float = Field(0.05, gt=0)
    sor: SORConfig = Field(default_factory=SORConfig)
    ror: RORConfig = Field(default_factory=RORConfig)
    normal_estimation: NormalEstimationConfig = Field(default_factory=NormalEstimationConfig)

class RansacConfig(BaseModel):
    """
    RANSAC (Random Sample Consensus) Hyperparameters.
    """
    distance_threshold: float = Field(0.02, gt=0) # Max distance from plane to be an inlier (metres)
    ransac_n: int = Field(3, ge=3)               # Min points to define a plane model (3 points = 1 plane)
    num_iterations: int = Field(2000, gt=0)      # How many times to try random samples
    min_plane_size: int = Field(1000, gt=0)      # Planes with fewer points are ignored
    max_planes: int = Field(10, gt=0)            # Maximum number of planes to extract
    remaining_points_min: int = Field(500, ge=0) # Stop if fewer than this many points remain

class DbscanConfig(BaseModel):
    eps: float = Field(0.1, gt=0)
    min_points: int = Field(50, gt=0)
    min_cluster_points: int = Field(100, gt=0)
    max_object_size: float = Field(3.0, gt=0)

class LabelingConfig(BaseModel):
    """
    Heuristics for semantic classification.
    """
    floor_z_threshold: float = Field(0.15)            # Planes below this Z are floors
    ceiling_z_fraction: float = Field(0.8, ge=0, le=1.0) # Planes above this fraction of max height are ceilings
    horizontal_angle_deg: float = Field(15, ge=0, le=90) # Max angle from Z-axis to be 'horizontal'
    vertical_angle_deg: float = Field(75, ge=0, le=90)   # Min angle from Z-axis to be 'vertical' (wall)
    tall_furniture_min_h: float = Field(1.5, gt=0)      # Objects taller than this are 'tall_furniture'
    furniture_min_footprint: float = Field(0.3, gt=0)   # Min area (W*D) to be considered furniture
    furniture_min_h: float = Field(0.3, gt=0)           # Min height to be considered furniture
    small_object_max_footprint: float = Field(0.5, gt=0)# Max area for 'small_object'
    high_fixture_min_z: float = Field(1.5, gt=0)        # Objects starting above this Z are 'high_fixtures'

class OutputConfig(BaseModel):
    ply: str = "outputs/segmented_room.ply"
    report: str = "outputs/segmentation_report.json"
    screenshot: str = "outputs/segmentation_viz.png"

class PipelineConfig(BaseModel):
    """Top-level configuration model for the entire pipeline."""
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    ransac: RansacConfig = Field(default_factory=RansacConfig)
    dbscan: DbscanConfig = Field(default_factory=DbscanConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


# ── Plane result ─────────────────────────────────────────────────────────────

class PlaneResult(BaseModel):
    """Represents a geometric plane detected via RANSAC with semantic labeling."""
    """
    Output of IterativeRANSAC for a single detected plane.
    """
    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, item):
        """Allows dictionary-like access for backward compatibility with tests."""
        return getattr(self, item)

    def __contains__(self, item):
        return hasattr(self, item)

    def keys(self):
        return self.model_fields.keys()

    plane_id: int = Field(..., ge=0, description="Sequential plane index.")
    label: str = Field("unknown", description="Semantic label (floor, wall, etc.).")
    inlier_count: int = Field(..., gt=0) # Number of points that fit this plane
    centroid_z: float                 # Average Z coordinate (used for floor/ceiling check)
    normal: List[float] = Field(..., min_length=3, max_length=3) # [nx, ny, nz] unit vector
    plane_model: List[float] = Field(..., min_length=4, max_length=4,
                                     description="[a, b, c, d] coefficients of ax+by+cz+d=0.")
    
    # Non-serializable cloud object
    inlier_cloud: Optional[o3d.geometry.PointCloud] = Field(None, exclude=True)

    @field_validator("normal")
    @classmethod
    def normal_not_all_zero(cls, v: List[float]) -> List[float]:
        if all(abs(x) < 1e-9 for x in v):
            raise ValueError("Normal vector must not be the zero vector.")
        return v

    @field_validator("label")
    @classmethod
    def label_is_known(cls, v: str) -> str:
        # INTERVIEW TIP: Why use a fixed set of labels?
        # Answer: Ensures consistency between labeling logic and visualization/exporting.
        # It prevents typos in labels from breaking the pipeline.
        valid = {
            "floor", "ceiling", "wall", "horizontal_surface",
            "unknown", "noise"
        }
        if v not in valid:
            raise ValueError(f"Plane label '{v}' is not a recognised semantic label. "
                             f"Valid options: {sorted(valid)}")
        return v


# ── Cluster result ────────────────────────────────────────────────────────────

class ClusterResult(BaseModel):
    """Represents a point cloud cluster detected via DBSCAN with geometric properties."""
    """
    Output of DBSCANClusterer + SemanticLabeler + BoundingBoxEstimator
    for a single detected object cluster.
    """
    model_config = {"arbitrary_types_allowed": True}

    def __getitem__(self, item):
        """Allows dictionary-like access for backward compatibility with tests."""
        if item == 'aabb':
            return getattr(self, 'aabb_box', None)
        return getattr(self, item)

    def __contains__(self, item):
        return hasattr(self, item)

    def keys(self):
        return self.model_fields.keys()

    cluster_id: int = Field(..., ge=0) # Unique ID for the object cluster
    label: str = Field("unknown")    # Semantic label (furniture, chair, etc.)
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Label confidence score.") # Probability [0, 1]
    n_points: int = Field(..., gt=0) # Number of points in this cluster
    point_density: float = Field(0.0, ge=0.0, description="Points per cubic metre.") # How packed the points are
    centroid: List[float] = Field(..., min_length=3, max_length=3) # [x, y, z] center of the object
    dims: List[float] = Field(..., min_length=3, max_length=3,
                               description="[width, depth, height] in metres.") # Physical size
    z_min: float # Lowest Z coordinate (floor contact)
    z_max: float # Highest Z coordinate
    footprint_m2: float = Field(..., ge=0.0) # Horizontal area (width * depth)

    # Optional — only present after BoundingBoxEstimator
    obb_extent: Optional[List[float]] = None
    obb_rotation_deg: Optional[float] = None
    
    # Non-serializable cloud object
    cloud: Optional[o3d.geometry.PointCloud] = Field(None, exclude=True)
    # Open3D BBox objects
    aabb_box: Optional[o3d.geometry.AxisAlignedBoundingBox] = Field(None, exclude=True)
    obb_box: Optional[o3d.geometry.OrientedBoundingBox] = Field(None, exclude=True)

    @model_validator(mode="after")
    def z_min_le_z_max(self) -> "ClusterResult":
        if self.z_min > self.z_max + 1e-6:
            raise ValueError(
                f"z_min ({self.z_min:.3f}) must be <= z_max ({self.z_max:.3f})."
            )
        return self

    @field_validator("dims")
    @classmethod
    def dims_non_negative(cls, v: List[float]) -> List[float]:
        if any(d < -1e-6 for d in v):
            raise ValueError(f"Bounding box dimensions must be non-negative, got {v}.")
        return v

    @field_validator("label")
    @classmethod
    def label_is_known(cls, v: str) -> str:
        valid = {
            "furniture", "tall_furniture", "small_object",
            "high_fixture", "horizontal_surface", "unknown", "noise",
            "chair", "table", "shelf"
        }
        if v not in valid:
            raise ValueError(f"Cluster label '{v}' is not a recognised semantic label. "
                             f"Valid options: {sorted(valid)}")
        return v


# ── Segmentation report ───────────────────────────────────────────────────────

class SegmentationReport(BaseModel):
    """Top-level JSON report structure (mirrors Exporter output for validation)."""
    structural_planes: List[PlaneResult] = Field(default_factory=list)
    objects: List[ClusterResult] = Field(default_factory=list)

    @property
    def plane_label_counts(self) -> dict:
        counts: dict = {}
        for p in self.structural_planes:
            counts[p.label] = counts.get(p.label, 0) + 1
        return counts

    @property
    def object_label_counts(self) -> dict:
        counts: dict = {}
        for c in self.objects:
            counts[c.label] = counts.get(c.label, 0) + 1
        return counts
