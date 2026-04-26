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
            cloud = cluster.cloud
            if cloud is None or len(cloud.points) == 0:
                logger.warning(f"  [Skipping] Cluster {cluster.cluster_id} has 0 points.")
                continue

            # ── AABB ──────────────────────────────────────────────────────
            aabb = cloud.get_axis_aligned_bounding_box()
            aabb.color = (1.0, 0.0, 0.0)
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)
            cluster.aabb_box = aabb
            cluster.dims = dims.tolist()

            # ── OBB ──────────────────────────────────────────────────────
            try:
                if len(cloud.points) >= 4:
                    obb = cloud.get_oriented_bounding_box()
                    obb.color = (0.0, 1.0, 0.0)
                    R = np.asarray(obb.R)
                    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                    cluster.obb_box = obb
                    cluster.obb_extent = obb.extent.tolist()
                    cluster.obb_rotation_deg = yaw_deg
                else:
                    cluster.obb_box = None
                    cluster.obb_extent = dims.tolist()
                    cluster.obb_rotation_deg = 0.0
            except Exception:
                cluster.obb_box = None
                cluster.obb_extent = dims.tolist()
                cluster.obb_rotation_deg = 0.0

            logger.info(
                f"  [{cluster.label}] id={cluster.cluster_id} "
                f"W={dims[0]:.2f}m D={dims[1]:.2f}m H={dims[2]:.2f}m "
                f"yaw={cluster.obb_rotation_deg:.1f}°"
            )

        return clusters
