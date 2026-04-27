import open3d as o3d
import numpy as np
import logging
from typing import List, Union
from src.models import ClusterResult, DbscanConfig

logger = logging.getLogger(__name__)


class DBSCANClusterer:
    """
    Module 4: Residual Clustering (DBSCAN)
    Groups remaining points into object-like clusters.
    
    INTERVIEW TIP: Why DBSCAN over K-Means?
    1. Doesn't require knowing the number of clusters (K) beforehand.
    2. Can find clusters of arbitrary shapes.
    3. Naturally handles noise (outliers).
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
        # Guard clause: If there are fewer than 2 points, clustering is impossible.
        if len(residual_pcd.points) < 2:
            return []

        # Extracting parameters from the configuration object for local use.
        eps = self.cfg.eps
        min_points = self.cfg.min_points
        min_cluster_points = self.cfg.min_cluster_points
        max_object_size = self.cfg.max_object_size

        # cluster_dbscan is a built-in Open3D function that performs Density-Based Clustering.
        # It returns an array of integers where each integer is the cluster label for that point.
        # A label of -1 means the point is noise.
        labels = np.array(
            residual_pcd.cluster_dbscan(
                eps=eps, min_points=min_points, print_progress=False
            )
        )

        # If no labels were returned or all labels are -1 (noise), return an empty list.
        if labels.size == 0 or labels.max() < 0:
            logger.info("DBSCAN: no clusters found (all points are noise).")
            return []

        # Find the maximum label index to know how many clusters were found.
        max_label = int(labels.max())
        logger.debug(f"DBSCAN found {max_label + 1} raw cluster(s) (before filtering).")

        # Initialize an empty list to store the final validated ClusterResult objects.
        clusters: List[ClusterResult] = []
        
        # Iterate through every detected cluster index.
        for i in range(max_label + 1):
            # Find the indices of all points that belong to the current cluster 'i'.
            indices = np.where(labels == i)[0]

            # If the cluster is smaller than our minimum point threshold, ignore it.
            if indices.size < min_cluster_points:
                continue

            # Extract the points for this specific cluster into a new PointCloud object.
            cluster_cloud = residual_pcd.select_by_index(indices)
            # Convert points to a numpy array for statistical calculations.
            pts = np.asarray(cluster_cloud.points)

            # Get the Axis-Aligned Bounding Box (AABB) to check physical dimensions.
            aabb = cluster_cloud.get_axis_aligned_bounding_box()
            # Calculate the width, depth, and height [dx, dy, dz].
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)

            # If the cluster is physically too large (e.g. wall fragments or merged objects), skip it.
            if np.any(dims > max_object_size):
                continue

            # Calculate the average position (centroid) of the points.
            centroid = pts.mean(axis=0)
            # Find the lowest and highest Z-coordinate for semantic labeling later.
            z_min = float(pts[:, 2].min())
            z_max = float(pts[:, 2].max())

            # Append a new ClusterResult object with all calculated metadata.
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
                    footprint_m2=float(dims[0] * dims[1]), # Area covered on the floor plane.
                )
            )

        # Sort the clusters so the largest ones (by point count) come first.
        clusters.sort(key=lambda c: c.n_points, reverse=True)
        logger.info(f"Retained {len(clusters)} cluster(s) after size filtering.")
        return clusters
