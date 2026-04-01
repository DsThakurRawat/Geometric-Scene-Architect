"""
src/models.py — Pydantic data models for the 3D segmentation pipeline.

Using Pydantic v2 for:
  - Runtime type validation of pipeline outputs
  - Clean JSON serialization (report, logs)
  - Catching dict key errors at construction time rather than access time
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Plane result ─────────────────────────────────────────────────────────────

class PlaneResult(BaseModel):
    """
    Output of IterativeRANSAC for a single detected plane.
    The `inlier_cloud` is NOT stored in the Pydantic model (it's an Open3D object);
    this model captures only the serializable metadata.
    """
    model_config = {"arbitrary_types_allowed": True}

    plane_id: int = Field(..., ge=0, description="Sequential plane index.")
    label: str = Field("unknown", description="Semantic label assigned by SemanticLabeler.")
    inlier_count: int = Field(..., gt=0)
    centroid_z: float
    normal: List[float] = Field(..., min_length=3, max_length=3)
    plane_model: List[float] = Field(..., min_length=4, max_length=4,
                                     description="[a, b, c, d] coefficients of ax+by+cz+d=0.")

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
    """
    Output of DBSCANClusterer + SemanticLabeler + BoundingBoxEstimator
    for a single detected object cluster.
    `cloud` is NOT stored here — only serializable metadata.
    """
    model_config = {"arbitrary_types_allowed": True}

    cluster_id: int = Field(..., ge=0)
    label: str = Field("unknown")
    n_points: int = Field(..., gt=0)
    centroid: List[float] = Field(..., min_length=3, max_length=3)
    dims: List[float] = Field(..., min_length=3, max_length=3,
                               description="[width, depth, height] in metres.")
    z_min: float
    z_max: float
    footprint_m2: float = Field(..., ge=0.0)

    # Optional — only present after BoundingBoxEstimator
    obb_extent: Optional[List[float]] = None
    obb_rotation_deg: Optional[float] = None

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
            "high_fixture", "horizontal_surface", "unknown", "noise"
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
