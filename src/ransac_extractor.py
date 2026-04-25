import open3d as o3d
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class IterativeRANSAC:
    """
    Module 3: Planar Primitive Extraction (RANSAC)
    Iteratively detects and removes the largest planes (floor, ceiling, walls).
    Uses connected-component filtering (post-RANSAC DBSCAN) to reject
    coplanar-but-disconnected fragments.
    """

    def __init__(self, config: Dict):
        self.cfg = config.get("ransac", {})

    def extract_planes(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[List[Dict], o3d.geometry.PointCloud]:
        """
        Runs RANSAC in a loop to extract planes until no large planes remain.

        Returns:
            planes        – list of PlaneResult dicts.
            residual_pcd  – remaining cloud (potential furniture / clutter).
        """
        if len(pcd.points) == 0:
            return [], pcd

        remaining_pcd = pcd
        planes: List[Dict] = []

        dist_threshold = float(self.cfg.get("distance_threshold", 0.02))
        ransac_n = int(self.cfg.get("ransac_n", 3))
        num_iterations = int(self.cfg.get("num_iterations", 2000))
        min_plane_size = int(self.cfg.get("min_plane_size", 1000))
        max_planes = int(self.cfg.get("max_planes", 10))
        remaining_points_min = int(self.cfg.get("remaining_points_min", 500))

        for i in range(max_planes):
            if len(remaining_pcd.points) < max(remaining_points_min, ransac_n):
                break

            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=dist_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations,
            )

            if len(inliers) < min_plane_size:
                break

            inlier_cloud = remaining_pcd.select_by_index(inliers)

            # ── Post-RANSAC refinement: keep only the largest connected component ──
            # This prevents a far-away coplanar surface from being merged into the plane.
            refined_cloud = self._keep_largest_component(
                inlier_cloud, eps=dist_threshold * 5, min_points=10
            )

            # Remove original inliers from the remaining cloud so the loop terminates
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

            pts = np.asarray(refined_cloud.points)
            if len(pts) == 0:
                continue  # Degenerate refinement result — skip

            planes.append(
                {
                    "plane_model": plane_model,
                    "inlier_cloud": refined_cloud,
                    "inlier_count": len(pts),
                    "normal": list(plane_model[:3]),  # [a, b, c]
                    "centroid_z": float(np.median(pts[:, 2])),
                }
            )

            logger.info(
                f"  Plane {i + 1}: {len(pts)} pts | "
                f"normal=[{plane_model[0]:.3f}, {plane_model[1]:.3f}, {plane_model[2]:.3f}]"
            )

        return planes, remaining_pcd

    @staticmethod
    def _keep_largest_component(
        pcd: o3d.geometry.PointCloud, eps: float, min_points: int
    ) -> o3d.geometry.PointCloud:
        """
        Runs DBSCAN on the given cloud and returns only the largest cluster.
        Falls back to the original cloud if no valid cluster is found.
        """
        if len(pcd.points) < min_points:
            return pcd

        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        valid_mask = labels >= 0

        if not valid_mask.any():
            # All points are noise — return original as fallback
            return pcd

        valid_labels = labels[valid_mask]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        best_label = unique_labels[np.argmax(counts)]
        best_indices = np.where(labels == best_label)[0]
        return pcd.select_by_index(best_indices)
