import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class BoundingBoxEstimator:
    """
    Module 6: Bounding Boxes & Furniture Dimensions (Extra Credit)
    Computes Axis-Aligned (AABB) and Oriented Bounding Boxes (OBB) for each cluster.
    """

    def compute(self, clusters: List[Dict]) -> List[Dict]:
        """
        Annotates each cluster dict in-place with AABB and OBB geometry objects
        and their numeric summaries.  Safe to call on degenerate / near-planar clouds.
        """
        for cluster in clusters:
            cloud = cluster["cloud"]

            # ── AABB (always succeeds) ─────────────────────────────────────
            aabb = cloud.get_axis_aligned_bounding_box()
            aabb.color = (1.0, 0.0, 0.0)  # Red wireframe
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)
            cluster["aabb_box"] = aabb
            cluster["dims"] = dims.tolist()

            # ── OBB (may fail on coplanar or very small clouds) ────────────
            try:
                if len(cloud.points) >= 4:
                    obb = cloud.get_oriented_bounding_box()
                    obb.color = (0.0, 1.0, 0.0)  # Green wireframe
                    R = np.asarray(obb.R)
                    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                    cluster["obb_box"] = obb
                    cluster["obb_extent"] = obb.extent.tolist()
                    cluster["obb_rotation_deg"] = yaw_deg
                else:
                    cluster["obb_box"] = None
                    cluster["obb_extent"] = dims.tolist()
                    cluster["obb_rotation_deg"] = 0.0
            except Exception:
                # Singular covariance (planar / degenerate cluster) — fall back
                cluster["obb_box"] = None
                cluster["obb_extent"] = dims.tolist()
                cluster["obb_rotation_deg"] = 0.0

            logger.debug(
                f"  [{cluster.get('label', '?')}] id={cluster['label_id']} "
                f"W={dims[0]:.2f}m D={dims[1]:.2f}m H={dims[2]:.2f}m "
                f"yaw={cluster['obb_rotation_deg']:.1f}°"
            )

        return clusters
