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
        pts = np.asarray(pcd.points)
        n_pts = len(pts)

        if n_pts < 10:
            return [], pcd

        # Ensure normals exist
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )

        normals = np.asarray(pcd.normals)

        # ── Parameters ────────────────────────────────────────────────────
        angle_threshold = float(self.cfg.get("angle_threshold_deg", 10.0))
        curvature_threshold = float(self.cfg.get("curvature_threshold", 0.05))
        min_region_size = int(self.cfg.get("min_region_size", 100))
        max_regions = int(self.cfg.get("max_regions", 20))
        k_neighbours = int(self.cfg.get("k_neighbours", 30))

        cos_threshold = np.cos(np.radians(angle_threshold))

        # ── Curvature estimation (approximated by normal variation) ──────
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        curvatures = np.zeros(n_pts)

        for i in range(n_pts):
            _, idx, _ = kd_tree.search_knn_vector_3d(pts[i], k_neighbours)
            if len(idx) < 2:
                curvatures[i] = 1.0
                continue
            neighbour_normals = normals[idx[1:]]
            # Curvature ≈ variance of normal dot products with seed normal
            dots = np.abs(neighbour_normals @ normals[i])
            curvatures[i] = 1.0 - np.mean(dots)

        # Sort by curvature (start from flattest points)
        sorted_indices = np.argsort(curvatures)

        # ── Region growing ────────────────────────────────────────────────
        assigned = np.zeros(n_pts, dtype=bool)
        regions: List[List[int]] = []

        for start_idx in sorted_indices:
            if assigned[start_idx]:
                continue
            if len(regions) >= max_regions:
                break

            # Start a new region from this seed
            region: List[int] = []
            seeds = [int(start_idx)]
            assigned[start_idx] = True

            while seeds:
                seed = seeds.pop(0)
                region.append(seed)

                _, idx, _ = kd_tree.search_knn_vector_3d(pts[seed], k_neighbours)

                for neighbour in idx[1:]:
                    if assigned[neighbour]:
                        continue

                    # Check normal similarity
                    dot = abs(float(np.dot(normals[seed], normals[neighbour])))
                    if dot >= cos_threshold:
                        assigned[neighbour] = True
                        region.append(neighbour)

                        # If this neighbour is also flat, use it as a new seed
                        if curvatures[neighbour] < curvature_threshold:
                            seeds.append(int(neighbour))

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
