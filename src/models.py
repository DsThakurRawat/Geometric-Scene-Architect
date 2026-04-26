import open3d as o3d
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Configuration Models ─────────────────────────────────────────────────────

class SORConfig(BaseModel):
    nb_neighbors: int = Field(20, gt=0)
    std_ratio: float = Field(2.0, gt=0)

class NormalEstimationConfig(BaseModel):
    radius: float = Field(0.1, gt=0)
    max_nn: int = Field(30, gt=0)
    orient_k: int = Field(15, gt=0)

class PreprocessingConfig(BaseModel):
    voxel_size: float = Field(0.05, gt=0)
    sor: SORConfig = Field(default_factory=SORConfig)
    normal_estimation: NormalEstimationConfig = Field(default_factory=NormalEstimationConfig)

class RansacConfig(BaseModel):
    distance_threshold: float = Field(0.02, gt=0)
    ransac_n: int = Field(3, ge=3)
    num_iterations: int = Field(2000, gt=0)
    min_plane_size: int = Field(1000, gt=0)
    max_planes: int = Field(10, gt=0)
    remaining_points_min: int = Field(500, ge=0)

class DbscanConfig(BaseModel):
    eps: float = Field(0.1, gt=0)
    min_points: int = Field(50, gt=0)
    min_cluster_points: int = Field(100, gt=0)
    max_object_size: float = Field(3.0, gt=0)

class LabelingConfig(BaseModel):
    floor_z_threshold: float = Field(0.15)
    ceiling_z_fraction: float = Field(0.8, ge=0, le=1.0)
    horizontal_angle_deg: float = Field(15, ge=0, le=90)
    vertical_angle_deg: float = Field(75, ge=0, le=90)
    tall_furniture_min_h: float = Field(1.5, gt=0)
    furniture_min_footprint: float = Field(0.3, gt=0)
    furniture_min_h: float = Field(0.3, gt=0)
    small_object_max_footprint: float = Field(0.5, gt=0)
    high_fixture_min_z: float = Field(1.5, gt=0)

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
    label: str = Field("unknown", description="Semantic label assigned by SemanticLabeler.")
    inlier_count: int = Field(..., gt=0)
    centroid_z: float
    normal: List[float] = Field(..., min_length=3, max_length=3)
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

    cluster_id: int = Field(..., ge=0)
    label: str = Field("unknown")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Label confidence score.")
    n_points: int = Field(..., gt=0)
    point_density: float = Field(0.0, ge=0.0, description="Points per cubic metre.")
    centroid: List[float] = Field(..., min_length=3, max_length=3)
    dims: List[float] = Field(..., min_length=3, max_length=3,
                               description="[width, depth, height] in metres.")
    z_min: float
    z_max: float
    footprint_m2: float = Field(..., ge=0.0)

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
