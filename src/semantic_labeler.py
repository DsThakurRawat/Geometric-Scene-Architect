import numpy as np
import math
import logging
from typing import Dict, List, Union
from src.models import PlaneResult, ClusterResult, LabelingConfig

logger = logging.getLogger(__name__)

# Canonical semantic color map (RGB 0-1). Single source of truth for the whole project.
LABEL_COLORS: Dict[str, List[float]] = {
    "floor":              [0.6, 0.4, 0.2],   # Brown
    "ceiling":            [0.9, 0.9, 0.9],   # Light Gray
    "wall":               [0.5, 0.6, 0.8],   # Steel Blue
    "furniture":          [0.2, 0.8, 0.3],   # Green
    "chair":              [0.3, 0.9, 0.3],   # Bright Green
    "table":              [0.1, 0.6, 0.1],   # Dark Green
    "shelf":              [0.4, 0.4, 0.1],   # Olive
    "tall_furniture":     [0.1, 0.4, 0.1],   # Very Dark Green
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
    
    INTERVIEW TIP: Why Rule-Based vs Machine Learning?
    Pros: No training data needed, fast, interpretable, deterministic.
    Cons: Doesn't handle complex geometries well, requires manual tuning of thresholds.
    """

    COLOR_MAP = LABEL_COLORS

    def __init__(self, config: Union[LabelingConfig, dict]):
        if isinstance(config, dict):
            self.cfg = LabelingConfig(**config.get("labeling", {}))
        else:
            self.cfg = config

    # ── Plane labeling ────────────────────────────────────────────────────

    @staticmethod
    def _plane_geom(plane):
        """Return (normal_z_abs, angle_from_vertical_deg, centroid_z) or None if the normal is degenerate."""
        normal = np.array(
            getattr(plane, 'normal', [0, 0, 0]) if not isinstance(plane, dict) else plane.get('normal', [0, 0, 0]),
            dtype=float,
        )
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            return None
        nz = abs(float(normal[2] / norm_len))
        angle_from_vertical = math.degrees(math.acos(min(nz, 1.0)))
        centroid_z = float(getattr(plane, 'centroid_z', 0.0) if not isinstance(plane, dict) else plane.get('centroid_z', 0.0))
        return nz, angle_from_vertical, centroid_z

    def label_planes(self, planes: List[PlaneResult], scene_height: float) -> List[PlaneResult]:
        """Labels structural planes (floor / ceiling / wall / unknown).

        Ceiling detection anchors on the TOPMOST horizontal plane in the scene rather than a
        fixed fraction of the raw z-extent. The old `centroid_z > scene_height * ceiling_z_fraction`
        rule silently demoted a real, flat ceiling to `horizontal_surface` whenever anything above it
        (ducts, beams, a sloped roof, high fixtures) inflated `scene_height` — e.g. an S3DIS hallway
        whose 3.05 m ceiling fell just under 0.8 x 3.93 m. Anchoring on the highest horizontal plane
        is robust to that overhead clutter; `min_ceiling_z` keeps low tables from ever being promoted.
        """
        floor_z_thr   = self.cfg.floor_z_threshold
        ceil_z_frac   = self.cfg.ceiling_z_fraction
        ceiling_band  = self.cfg.ceiling_band_m
        min_ceiling_z = self.cfg.min_ceiling_z
        horiz_ang_thr = self.cfg.horizontal_angle_deg
        vert_ang_thr  = self.cfg.vertical_angle_deg

        effective_height = max(scene_height, 0.1)

        # ── pre-pass: the ceiling reference is the highest horizontal plane above the floor ──
        horizontal_zs = []
        for plane in planes:
            g = self._plane_geom(plane)
            if g is None:
                continue
            _, angle_from_vertical, centroid_z = g
            if angle_from_vertical < horiz_ang_thr and centroid_z >= floor_z_thr:
                horizontal_zs.append(centroid_z)
        # Fall back to the legacy fraction rule only when no horizontal plane was found at all.
        ceiling_ref = max(horizontal_zs) if horizontal_zs else effective_height * ceil_z_frac

        # ── main pass ──
        for plane in planes:
            g = self._plane_geom(plane)
            if g is None:
                # Degenerate normal — orientation is undefined.
                if not isinstance(plane, dict): plane.label = "unknown"
                else: plane["label"] = "unknown"
                inlier_cloud = getattr(plane, 'inlier_cloud', None) if not isinstance(plane, dict) else plane.get('inlier_cloud')
                if inlier_cloud:
                    inlier_cloud.paint_uniform_color(LABEL_COLORS["unknown"])
                continue

            nz, angle_from_vertical, centroid_z = g

            label = "unknown"
            # If the angle is small, the plane is horizontal (floor / ceiling / table-top).
            if angle_from_vertical < horiz_ang_thr:
                if centroid_z < floor_z_thr:
                    label = "floor"
                # Ceiling = a horizontal plane near the topmost horizontal plane, and plausibly high.
                elif centroid_z >= ceiling_ref - ceiling_band and centroid_z >= min_ceiling_z:
                    label = "ceiling"
                else:
                    label = "horizontal_surface"
            # If the angle is large, the plane is vertical (a wall).
            elif angle_from_vertical > vert_ang_thr:
                label = "wall"
            # Otherwise, it's some slanted surface we don't recognize.
            else:
                label = "unknown"

            # Assign the determined label to the plane object.
            if not isinstance(plane, dict): plane.label = label
            else: plane["label"] = label

            # Paint the points belonging to this plane with the canonical label color.
            inlier_cloud = getattr(plane, 'inlier_cloud', None) if not isinstance(plane, dict) else plane.get('inlier_cloud')
            if inlier_cloud:
                color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
                inlier_cloud.paint_uniform_color(color)

        return planes

    # ── Cluster labeling ──────────────────────────────────────────────────

    def label_clusters(self, clusters: List[ClusterResult]) -> List[ClusterResult]:
        """Labels object clusters using height, aspect ratio, and footprint heuristics."""
        # Pull threshold values from the config for use in the labeling logic.
        tall_h      = self.cfg.tall_furniture_min_h
        min_foot    = self.cfg.furniture_min_footprint
        min_h       = self.cfg.furniture_min_h
        max_small_f = self.cfg.small_object_max_footprint
        high_z      = self.cfg.high_fixture_min_z

        # Iterate through each detected cluster.
        for cluster in clusters:
            # Extract dimensions (width, depth, height).
            dims = getattr(cluster, 'dims', None) if not isinstance(cluster, dict) else cluster.get('dims')
            if dims is None: dims = [0, 0, 0]
            w, d, h = dims[0], dims[1], dims[2]
            # Get the starting Z-coordinate (the bottom of the object).
            z_base = float(getattr(cluster, 'z_min', 0.0) if not isinstance(cluster, dict) else cluster.get('z_min', 0.0))
            # Area covered by the object.
            footprint_m2 = float(getattr(cluster, 'footprint_m2', 0.0) if not isinstance(cluster, dict) else cluster.get('footprint_m2', 0.0))
            # Ratio of width to depth to distinguish long objects from square ones.
            aspect_ratio = max(w, d) / min(w, d) if min(w, d) > 1e-3 else 1.0

            # Calculate volume and point density.
            volume = max(w * d * h, 1e-6)
            n_points = getattr(cluster, 'n_points', 1) if not isinstance(cluster, dict) else cluster.get('n_points', 1)
            density = round(n_points / volume, 1)
            
            # Store the density value in the cluster object.
            if not isinstance(cluster, dict): cluster.point_density = density
            else: cluster["point_density"] = density

            # Default values.
            label = "furniture"
            confidence = 0.5

            # SCENARIO A: The object is sitting on or near the floor.
            if z_base < 0.2:
                # TALL OBJECTS (e.g., wardrobes, bookshelves)
                if h > tall_h:
                    label = "tall_furniture"
                    confidence = min(h / tall_h, 1.0)
                # MEDIUM HEIGHT OBJECTS (e.g., chairs, tables)
                elif 0.4 <= h <= 1.1:
                    # Small footprint and square shape = Chair
                    if 0.3 <= footprint_m2 <= 0.8 and aspect_ratio < 1.5:
                        label = "chair"
                        confidence = 0.7
                    # Large footprint = Table
                    elif footprint_m2 > 0.8:
                        label = "table"
                        confidence = 0.8
                    # Other medium furniture
                    else:
                        label = "furniture"
                        confidence = 0.4
                # LOW OBJECTS (e.g., footstools, trash cans)
                elif h < 0.4:
                    label = "small_object"
                    confidence = 0.6
                else:
                    label = "furniture"
                    confidence = 0.3
            # SCENARIO B: The object is elevated (e.g., on a wall or counter).
            elif 0.2 <= z_base < high_z:
                # Long and narrow elevated object = Shelf
                if aspect_ratio > 3.0 and w > 1.0:
                    label = "shelf"
                    confidence = 0.7
                # Small elevated object = Small object (e.g., a cup on a table)
                elif footprint_m2 < max_small_f:
                    label = "small_object"
                    confidence = 1.0 - (footprint_m2 / max_small_f)
                else:
                    label = "furniture"
                    confidence = 0.5
            # SCENARIO C: The object is high up (e.g., ceiling lights, high vents).
            elif z_base >= high_z:
                label = "high_fixture"
                confidence = min(z_base / high_z, 1.0)

            # Boost confidence for dense clusters (more likely to be a real object than noise).
            if density > 50000:
                confidence = min(confidence + 0.1, 1.0)

            # Assign the final label and confidence score.
            if not isinstance(cluster, dict):
                cluster.label = label
                cluster.confidence = round(confidence, 2)
            else:
                cluster["label"] = label
                cluster["confidence"] = round(confidence, 2)
                
            # Paint the cluster points with the canonical label color.
            cloud = getattr(cluster, 'cloud', None) if not isinstance(cluster, dict) else cluster.get('cloud')
            if cloud:
                color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
                cloud.paint_uniform_color(color)

        return clusters

