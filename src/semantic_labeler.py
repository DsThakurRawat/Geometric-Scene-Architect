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

    def label_planes(self, planes: List[PlaneResult], scene_height: float) -> List[PlaneResult]:
        """Labels structural planes (floor / ceiling / wall / unknown)."""
        # Pull threshold values from the config for use in the labeling logic.
        floor_z_thr   = self.cfg.floor_z_threshold
        ceil_z_frac   = self.cfg.ceiling_z_fraction
        horiz_ang_thr = self.cfg.horizontal_angle_deg
        vert_ang_thr  = self.cfg.vertical_angle_deg

        # Ensure scene_height is positive to avoid division by zero.
        effective_height = max(scene_height, 0.1)

        # Iterate through each detected plane to assign a label.
        for plane in planes:
            # Extract the normal vector of the plane.
            normal = np.array(getattr(plane, 'normal', [0,0,0]) if not isinstance(plane, dict) else plane.get('normal', [0,0,0]), dtype=float)
            # Calculate the length (norm) of the vector.
            norm_len = np.linalg.norm(normal)
            
            # If the normal is zero or invalid, we can't determine orientation.
            if norm_len < 1e-9:
                if not isinstance(plane, dict): plane.label = "unknown"
                else: plane["label"] = "unknown"
                
                # Try to color the cloud purple to indicate 'unknown'.
                inlier_cloud = getattr(plane, 'inlier_cloud', None) if not isinstance(plane, dict) else plane.get('inlier_cloud')
                if inlier_cloud:
                    inlier_cloud.paint_uniform_color(LABEL_COLORS["unknown"])
                continue

            # Normalize the vector to have a length of 1.
            normal /= norm_len
            # Get the Z-component of the normal. 
            # If nz is 1.0, the plane is perfectly horizontal. If nz is 0.0, it's perfectly vertical.
            nz = abs(float(normal[2]))
            # Convert the vertical component into an angle in degrees relative to the vertical axis.
            angle_from_vertical = math.degrees(math.acos(min(nz, 1.0)))
            # Get the average height (Z) of the plane's points.
            centroid_z = float(getattr(plane, 'centroid_z', 0.0) if not isinstance(plane, dict) else plane.get('centroid_z', 0.0))

            # Default label is unknown.
            label = "unknown"
            # If the angle is small, the plane is horizontal (floor or ceiling).
            if angle_from_vertical < horiz_ang_thr:
                # If it's near the bottom of the scene, it's a floor.
                if centroid_z < floor_z_thr:
                    label = "floor"
                # If it's near the top of the scene, it's a ceiling.
                elif centroid_z > effective_height * ceil_z_frac:
                    label = "ceiling"
                # Otherwise, it's a horizontal surface like a table top.
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

