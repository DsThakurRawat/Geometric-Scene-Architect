import numpy as np
import math
from typing import Dict, List, Union
from src.models import PlaneResult, ClusterResult, LabelingConfig

# Canonical semantic color map (RGB 0-1). Single source of truth for the whole project.
LABEL_COLORS: Dict[str, List[float]] = {
    "floor":              [0.6, 0.4, 0.2],   # Brown
    "ceiling":            [0.9, 0.9, 0.9],   # Light Gray
    "wall":               [0.5, 0.6, 0.8],   # Steel Blue
    "furniture":          [0.2, 0.8, 0.3],   # Green
    "tall_furniture":     [0.1, 0.5, 0.1],   # Dark Green
    "small_object":       [0.9, 0.6, 0.1],   # Amber
    "high_fixture":       [0.9, 0.2, 0.2],   # Red
    "horizontal_surface": [0.4, 0.4, 0.1],   # Olive
    "unknown":            [0.6, 0.0, 0.6],   # Purple
    "noise":              [0.3, 0.3, 0.3],   # Dark Gray
}


class SemanticLabeler:
    """
    Module 5: Rule-Based Semantic Labeling.
    Assigns labels to RANSAC planes and DBSCAN clusters using pure geometry.
    """

    # Expose for tests and other modules
    COLOR_MAP = LABEL_COLORS

    def __init__(self, config: Union[LabelingConfig, dict]):
        if isinstance(config, dict):
            self.cfg = LabelingConfig(**config.get("labeling", {}))
        else:
            self.cfg = config

    # ── Plane labeling ────────────────────────────────────────────────────

    def label_planes(self, planes: List[PlaneResult], scene_height: float) -> List[PlaneResult]:
        """Labels structural planes (floor / ceiling / wall / unknown)."""
        floor_z_thr   = self.cfg.floor_z_threshold
        ceil_z_frac   = self.cfg.ceiling_z_fraction
        horiz_ang_thr = self.cfg.horizontal_angle_deg
        vert_ang_thr  = self.cfg.vertical_angle_deg

        # Guard: degenerate scene height
        effective_height = max(scene_height, 0.1)

        for plane in planes:
            normal = np.array(plane.normal, dtype=float)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                plane.label = "unknown"
                if plane.inlier_cloud:
                    plane.inlier_cloud.paint_uniform_color(LABEL_COLORS["unknown"])
                continue

            normal /= norm_len
            nz  = abs(float(normal[2]))
            # Clamp to [0,1] to prevent domain errors from floating-point drift
            angle_from_vertical = math.degrees(math.acos(min(nz, 1.0)))

            centroid_z = float(plane.centroid_z)

            if angle_from_vertical < horiz_ang_thr:
                if centroid_z < floor_z_thr:
                    plane.label = "floor"
                elif centroid_z > effective_height * ceil_z_frac:
                    plane.label = "ceiling"
                else:
                    plane.label = "horizontal_surface"
            elif angle_from_vertical > vert_ang_thr:
                plane.label = "wall"
            else:
                plane.label = "unknown"

            if plane.inlier_cloud:
                color = LABEL_COLORS.get(plane.label, LABEL_COLORS["unknown"])
                plane.inlier_cloud.paint_uniform_color(color)

        return planes

    # ── Cluster labeling ──────────────────────────────────────────────────

    def label_clusters(self, clusters: List[ClusterResult]) -> List[ClusterResult]:
        """Labels object clusters using height, reach, and footprint heuristics."""
        tall_h      = self.cfg.tall_furniture_min_h
        min_foot    = self.cfg.furniture_min_footprint
        min_h       = self.cfg.furniture_min_h
        max_small_f = self.cfg.small_object_max_footprint
        high_z      = self.cfg.high_fixture_min_z

        for cluster in clusters:
            h_cluster    = float(cluster.z_max) - float(cluster.z_min)
            z_base       = float(cluster.z_min)
            footprint_m2 = float(cluster.footprint_m2)

            if z_base < 0.15:
                if h_cluster > tall_h:
                    label = "tall_furniture"
                elif footprint_m2 > min_foot and h_cluster > min_h:
                    label = "furniture"
                else:
                    label = "furniture"  # small floor-level object
            elif 0.15 <= z_base < high_z:
                if footprint_m2 < max_small_f:
                    label = "small_object"
                else:
                    label = "furniture"
            elif z_base >= high_z:
                label = "high_fixture"
            else:
                label = "furniture"

            cluster.label = label
            if cluster.cloud:
                color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
                cluster.cloud.paint_uniform_color(color)

        return clusters
