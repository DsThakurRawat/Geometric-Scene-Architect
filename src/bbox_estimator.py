import numpy as np
import logging
from typing import List, Union
from src.models import PlaneResult, ClusterResult

logger = logging.getLogger(__name__)


class BoundingBoxEstimator:
    """
    Module 6: Bounding Boxes & Furniture Dimensions
    Computes AABB and OBB for each cluster.
    """

    def compute(self, clusters: List[ClusterResult]) -> List[ClusterResult]:
        """Annotates each cluster with AABB and OBB geometry and summaries."""
        for cluster in clusters:
            # Handle both models and dicts
            cloud = getattr(cluster, 'cloud', None) if not isinstance(cluster, dict) else cluster.get('cloud')
            if cloud is None or len(cloud.points) == 0:
                cluster_id = getattr(cluster, 'cluster_id', 'unknown') if not isinstance(cluster, dict) else cluster.get('cluster_id', 'unknown')
                logger.warning(f"  [Skipping] Cluster {cluster_id} has 0 points.")
                continue

            # ── AABB ──────────────────────────────────────────────────────
            aabb = cloud.get_axis_aligned_bounding_box()
            aabb.color = (1.0, 0.0, 0.0)
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)
            
            if not isinstance(cluster, dict):
                cluster.aabb_box = aabb
                cluster.dims = dims.tolist()
            else:
                cluster["aabb_box"] = aabb
                cluster["dims"] = dims.tolist()

            # ── OBB ──────────────────────────────────────────────────────
            obb_box, obb_extent, obb_rotation_deg = None, dims.tolist(), 0.0
            try:
                if len(cloud.points) >= 4:
                    obb = cloud.get_oriented_bounding_box()
                    obb.color = (0.0, 1.0, 0.0)
                    R = np.asarray(obb.R)
                    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                    obb_box, obb_extent, obb_rotation_deg = obb, obb.extent.tolist(), yaw_deg
            except Exception:
                pass

            if not isinstance(cluster, dict):
                cluster.obb_box = obb_box
                cluster.obb_extent = obb_extent
                cluster.obb_rotation_deg = obb_rotation_deg
            else:
                cluster["obb_box"] = obb_box
                cluster["obb_extent"] = obb_extent
                cluster["obb_rotation_deg"] = obb_rotation_deg

            label = getattr(cluster, 'label', 'unknown') if not isinstance(cluster, dict) else cluster.get('label', 'unknown')
            cluster_id = getattr(cluster, 'cluster_id', 'unknown') if not isinstance(cluster, dict) else cluster.get('cluster_id', 'unknown')
            logger.info(
                f"  [{label}] id={cluster_id} "
                f"W={dims[0]:.2f}m D={dims[1]:.2f}m H={dims[2]:.2f}m "
                f"yaw={obb_rotation_deg:.1f}°"
            )

        return clusters
