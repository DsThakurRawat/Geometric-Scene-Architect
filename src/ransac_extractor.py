import open3d as o3d
import numpy as np
import logging
from typing import List, Tuple, Union
from src.models import PlaneResult, RansacConfig

logger = logging.getLogger(__name__)


class IterativeRANSAC:
    """
    Module 3: Planar Primitive Extraction (RANSAC)
    Iteratively detects and removes the largest planes (floor, ceiling, walls).
    """

    def __init__(self, config: Union[RansacConfig, dict]):
        if isinstance(config, dict):
            self.cfg = RansacConfig(**config.get("ransac", {}))
        else:
            self.cfg = config

    def extract_planes(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[List[PlaneResult], o3d.geometry.PointCloud]:
        """
        Runs RANSAC in a loop to extract planes until no large planes remain.

        Returns:
            planes        – list of PlaneResult objects.
            residual_pcd  – remaining cloud (potential furniture / clutter).
        """
        if len(pcd.points) == 0:
            return [], pcd

        remaining_pcd = pcd
        planes: List[PlaneResult] = []

        dist_threshold = self.cfg.distance_threshold
        ransac_n = self.cfg.ransac_n
        num_iterations = self.cfg.num_iterations
        min_plane_size = self.cfg.min_plane_size
        max_planes = self.cfg.max_planes
        remaining_points_min = self.cfg.remaining_points_min

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
            refined_cloud = self._keep_largest_component(
                inlier_cloud, eps=dist_threshold * 5, min_points=10
            )
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

            pts = np.asarray(refined_cloud.points)
            if len(pts) == 0:
                continue

            planes.append(
                PlaneResult(
                    plane_id=i,
                    plane_model=plane_model.tolist(),
                    inlier_cloud=refined_cloud,
                    inlier_count=len(pts),
                    normal=plane_model[:3].tolist(),
                    centroid_z=float(np.median(pts[:, 2])),
                )
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
        """Runs DBSCAN and returns only the largest cluster."""
        if len(pcd.points) < min_points:
            return pcd
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        valid_mask = labels >= 0
        if not valid_mask.any():
            return pcd
        valid_labels = labels[valid_mask]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        best_label = unique_labels[np.argmax(counts)]
        best_indices = np.where(labels == best_label)[0]
        return pcd.select_by_index(best_indices)
