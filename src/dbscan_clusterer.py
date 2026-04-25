import open3d as o3d
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DBSCANClusterer:
    """
    Module 4: Residual Clustering (DBSCAN)
    Groups remaining points into object-like clusters.
    """

    def __init__(self, config: Dict):
        self.cfg = config.get("dbscan", {})

    def cluster(self, residual_pcd: o3d.geometry.PointCloud) -> List[Dict]:
        """
        Runs DBSCAN clustering and filters out noise / over-sized blobs.

        Returns:
            clusters – list of cluster-result dicts, sorted by n_points descending.
        """
        if len(residual_pcd.points) < 2:
            return []

        eps = float(self.cfg.get("eps", 0.10))
        min_points = int(self.cfg.get("min_points", 50))
        min_cluster_points = int(self.cfg.get("min_cluster_points", 100))
        max_object_size = float(self.cfg.get("max_object_size", 3.0))

        labels = np.array(
            residual_pcd.cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )

        if labels.size == 0 or labels.max() < 0:
            # All points classified as noise (-1)
            logger.info("DBSCAN: no clusters found (all points are noise).")
            return []

        max_label = int(labels.max())
        logger.debug(f"DBSCAN found {max_label + 1} raw cluster(s) (before filtering).")

        clusters: List[Dict] = []
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
                {
                    "label_id": i,
                    "cloud": cluster_cloud,
                    "n_points": int(indices.size),
                    "aabb": aabb,
                    "centroid": centroid.tolist(),
                    "dims": dims.tolist(),
                    "z_min": z_min,
                    "z_max": z_max,
                    "footprint_m2": float(dims[0] * dims[1]),
                }
            )

        # Sort by size descending so dominant objects come first
        clusters.sort(key=lambda c: c["n_points"], reverse=True)
        logger.info(f"Retained {len(clusters)} cluster(s) after size filtering.")
        return clusters
