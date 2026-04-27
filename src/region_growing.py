"""
src/region_growing.py — Normal-based Region Growing segmentation.

An alternative to RANSAC + DBSCAN that groups points with similar surface
normals into coherent regions. Useful for detecting smooth planar surfaces
and objects with consistent curvature.
"""
import open3d as o3d
import numpy as np
from typing import Dict, List, Tuple


class RegionGrowing:
    """
    Region Growing segmentation: groups neighbouring points whose normals
    differ by less than `angle_threshold_deg`.

    Algorithm:
        1. Start from the point with the smallest curvature (flattest region).
        2. Add neighbours whose normal angle difference is below the threshold.
        3. If a neighbour is also below the curvature threshold, add it as
           a new seed (allowing the region to expand further).
        4. Repeat until all seeds are exhausted, then move to the next
           un-assigned point.

    This is a pure-Python implementation using Open3D's KDTree for neighbour
    lookups. It is intentionally kept simple for readability.
    """

    def __init__(self, config: Dict):
        self.cfg = config.get("region_growing", {})

    def segment(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[List[Dict], o3d.geometry.PointCloud]:
        """
        Segments the point cloud into regions with consistent normals.

        Returns:
            regions      – list of region dicts (with 'cloud', 'n_points', etc.)
            residual_pcd – points not assigned to any region.
        """
        # Convert points to a NumPy array for fast calculations.
        pts = np.asarray(pcd.points)
        # Total number of points in the cloud.
        n_pts = len(pts)

        # If the cloud is too small, skip segmentation.
        if n_pts < 10:
            return [], pcd

        # Region Growing depends on surface normals. If they don't exist, compute them.
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        # Retrieve the computed normals.
        normals = np.asarray(pcd.normals)

        # ── Parameters ────────────────────────────────────────────────────
        angle_threshold = float(self.cfg.get("angle_threshold_deg", 10.0))
        curvature_threshold = float(self.cfg.get("curvature_threshold", 0.05))
        min_region_size = int(self.cfg.get("min_region_size", 100))
        max_regions = int(self.cfg.get("max_regions", 20))
        k_neighbours = int(self.cfg.get("k_neighbours", 30))

        cos_threshold = np.cos(np.radians(angle_threshold))

        # ── Curvature estimation (approximated by normal variation) ──────
        # Use a KDTree for fast proximity searches (finding neighbors).
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        # Initialize an array to store curvature values for every point.
        curvatures = np.zeros(n_pts)

        # Calculate curvature for each point by looking at the variation of normals in its neighborhood.
        for i in range(n_pts):
            # Find the k nearest neighbors for the current point.
            _, idx, _ = kd_tree.search_knn_vector_3d(pts[i], k_neighbours)
            # If no neighbors, assign maximum curvature.
            if len(idx) < 2:
                curvatures[i] = 1.0
                continue
            # Get normals of all neighbors.
            neighbour_normals = normals[idx[1:]]
            # Curvature ≈ variance of normal dot products with seed normal. 
            # High variance = high curvature (the surface is curving steeply).
            dots = np.abs(neighbour_normals @ normals[i])
            curvatures[i] = 1.0 - np.mean(dots)

        # Sort all points by curvature. We want to start growing regions from the flattest points first.
        sorted_indices = np.argsort(curvatures)

        # Keep track of which points have already been assigned to a region.
        assigned = np.zeros(n_pts, dtype=bool)
        # List to store the final grouped indices.
        regions: List[List[int]] = []

        # Start from the flatest point and try to grow a region.
        for start_idx in sorted_indices:
            # Skip if this point is already part of another region.
            if assigned[start_idx]:
                continue
            # Stop if we've reached the maximum allowed regions.
            if len(regions) >= max_regions:
                break

            # Start a new region list.
            region: List[int] = []
            # Use a 'seeds' queue to explore neighbors.
            seeds = [int(start_idx)]
            # Mark the starting point as assigned.
            assigned[start_idx] = True

            # Standard Breadth-First Search (BFS) for region growing.
            while seeds:
                # Take a point from the queue.
                seed = seeds.pop(0)
                # Add it to the current region.
                region.append(seed)

                # Find its neighbors.
                _, idx, _ = kd_tree.search_knn_vector_3d(pts[seed], k_neighbours)

                # Check each neighbor for similarity.
                for neighbour in idx[1:]:
                    # Skip if already assigned.
                    if assigned[neighbour]:
                        continue

                    # Check normal similarity using the dot product (cosine of the angle).
                    dot = abs(float(np.dot(normals[seed], normals[neighbour])))
                    # If the angle difference is small enough, the neighbor belongs to this region.
                    if dot >= cos_threshold:
                        # Mark as assigned and add to region.
                        assigned[neighbour] = True
                        region.append(neighbour)

                        # CRITICAL STEP: If this neighbor is also 'flat' enough, it becomes a new seed
                        # allowing the algorithm to 'flow' across a large planar surface.
                        if curvatures[neighbour] < curvature_threshold:
                            seeds.append(int(neighbour))

            # Only keep regions that meet the minimum size requirement.
            if len(region) >= min_region_size:
                regions.append(region)

        # ── Build output ──────────────────────────────────────────────────
        result: List[Dict] = []
        all_assigned_indices: set = set()

        for i, region_indices in enumerate(regions):
            region_cloud = pcd.select_by_index(region_indices)
            region_pts = np.asarray(region_cloud.points)
            centroid = region_pts.mean(axis=0)

            aabb = region_cloud.get_axis_aligned_bounding_box()
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)

            # Compute dominant normal
            region_normals = normals[region_indices]
            mean_normal = region_normals.mean(axis=0)
            norm_len = np.linalg.norm(mean_normal)
            if norm_len > 1e-9:
                mean_normal /= norm_len

            result.append({
                "region_id": i,
                "cloud": region_cloud,
                "n_points": len(region_indices),
                "centroid": centroid.tolist(),
                "centroid_z": float(np.median(region_pts[:, 2])),
                "dims": dims.tolist(),
                "normal": mean_normal.tolist(),
                "z_min": float(region_pts[:, 2].min()),
                "z_max": float(region_pts[:, 2].max()),
                "footprint_m2": float(dims[0] * dims[1]),
            })
            all_assigned_indices.update(region_indices)

        # ── Residual ──────────────────────────────────────────────────────
        residual_indices = [i for i in range(n_pts) if i not in all_assigned_indices]
        if residual_indices:
            residual_pcd = pcd.select_by_index(residual_indices)
        else:
            residual_pcd = o3d.geometry.PointCloud()

        return result, residual_pcd
