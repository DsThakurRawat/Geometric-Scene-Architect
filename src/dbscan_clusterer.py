import open3d as o3d
import numpy as np
from typing import List, Union
from src.models import ClusterResult, DbscanConfig


class DBSCANClusterer:
    """
    Module 4: Residual Clustering (DBSCAN)
    Groups remaining points into object-like clusters.
    """

    def __init__(self, config: Union[DbscanConfig, dict]):
        if isinstance(config, dict):
            self.cfg = DbscanConfig(**config.get("dbscan", {}))
        else:
            self.cfg = config

    def cluster(self, residual_pcd: o3d.geometry.PointCloud) -> List[ClusterResult]:
        """
        Runs DBSCAN clustering and filters out noise / over-sized blobs.

        Returns:
            clusters – list of ClusterResult objects, sorted by n_points descending.
        """
        if len(residual_pcd.points) < 2:
            return []

        eps = self.cfg.eps
        min_points = self.cfg.min_points
        min_cluster_points = self.cfg.min_cluster_points
        max_object_size = self.cfg.max_object_size

        labels = np.array(
            residual_pcd.cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )

        if labels.size == 0 or labels.max() < 0:
            # All points classified as noise (-1)
            print("DBSCAN: no clusters found (all points are noise).")
            return []

        max_label = int(labels.max())
        print(f"DBSCAN found {max_label + 1} raw cluster(s) (before filtering).")

        clusters: List[ClusterResult] = []
        for i in range(max_label + 1):
            indices = np.where(labels == i)[0]

            if indices.size < min_cluster_points:
                continue

            cluster_cloud = residual_pcd.select_by_index(indices)
            pts = np.asarray(cluster_cloud.points)

            aabb = cluster_cloud.get_axis_aligned_bounding_box()
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)

            # Reject blobs that are larger than any real piece of furniture
            if np.any(dims > max_object_size):
                continue

            centroid = pts.mean(axis=0)
            z_min = float(pts[:, 2].min())
            z_max = float(pts[:, 2].max())

            clusters.append(
                ClusterResult(
                    cluster_id=i,
                    cloud=cluster_cloud,
                    n_points=int(indices.size),
                    aabb_box=aabb,
                    centroid=centroid.tolist(),
                    dims=dims.tolist(),
                    z_min=z_min,
                    z_max=z_max,
                    footprint_m2=float(dims[0] * dims[1]),
                )
            )

        # Sort by size descending so dominant objects come first
        clusters.sort(key=lambda c: c.n_points, reverse=True)
        print(f"Retained {len(clusters)} cluster(s) after size filtering.")
        return clusters
