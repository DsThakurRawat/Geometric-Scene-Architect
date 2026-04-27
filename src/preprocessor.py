import open3d as o3d
from typing import Tuple, Union
from src.models import PreprocessingConfig


class Preprocessor:
    """
    Module 2: Preprocessing Pipeline
    Handles denoising, downsampling, and normal estimation.
    """

    def __init__(self, config: Union[PreprocessingConfig, dict]):
        self._raw_cfg = config
        self._validated_cfg = None if isinstance(config, dict) else config

    @property
    def cfg(self) -> PreprocessingConfig:
        """
        Lazy-loading property to get the validated configuration.
        Ensures that if a dictionary was passed to __init__, it gets converted to a Pydantic model.
        """
        if self._validated_cfg is None:
            from pydantic import ValidationError
            try:
                # Handle both wrapped and unwrapped dicts
                inner = self._raw_cfg.get("preprocessing", self._raw_cfg)
                self._validated_cfg = PreprocessingConfig(**inner)
            except ValidationError as e:
                raise ValueError(str(e))
        return self._validated_cfg

    def voxel_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Reduces point density using voxel downsampling.
        
        INTERVIEW TIP: Why downsample?
        1. Speed: Reduces the number of points for subsequent algorithms (RANSAC, DBSCAN).
        2. Uniformity: Ensures points are evenly distributed, preventing bias in dense areas.
        """
        voxel_size = self.cfg.voxel_size
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    def remove_statistical_outliers(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Removes floating scan noise via Statistical Outlier Removal (SOR).
        Returns (clean_pcd, outlier_pcd).
        
        INTERVIEW TIP: How does SOR work?
        It calculates the average distance of each point to its neighbors. 
        If the distance is significantly larger than the global average, it's an outlier.
        """
        nb_neighbors = self.cfg.sor.nb_neighbors
        std_ratio = self.cfg.sor.std_ratio

        empty = o3d.geometry.PointCloud()

        if len(pcd.points) <= nb_neighbors:
            # Too few points to compute statistics — return as-is
            return pcd, empty

        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        # Select the 'inlier' points (clean) and the 'outlier' points (noise)
        clean_pcd = pcd.select_by_index(ind)
        outlier_pcd = pcd.select_by_index(ind, invert=True)

        # Guard: if SOR removes everything (too aggressive), return original to avoid empty results
        if len(clean_pcd.points) == 0:
            return pcd, outlier_pcd

        return clean_pcd, outlier_pcd

    def remove_radius_outliers(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Removes isolated points via Radius Outlier Removal (ROR).
        A point is removed if fewer than `min_neighbors` points exist within `radius`.
        Returns (clean_pcd, outlier_pcd).
        
        INTERVIEW TIP: SOR vs ROR?
        SOR is better for sparse, global noise. 
        ROR is better for removing small, isolated clusters of noise.
        """
        radius = self.cfg.ror.radius
        min_neighbors = self.cfg.ror.min_neighbors

        empty = o3d.geometry.PointCloud()

        if len(pcd.points) < 2:
            return pcd, empty

        cl, ind = pcd.remove_radius_outlier(
            nb_points=min_neighbors, radius=radius
        )
        clean_pcd = pcd.select_by_index(ind)
        outlier_pcd = pcd.select_by_index(ind, invert=True)

        if len(clean_pcd.points) == 0:
            return pcd, outlier_pcd

        return clean_pcd, outlier_pcd

    def estimate_normals(self, pcd: o3d.geometry.PointCloud) -> None:
        """
        Estimates surface normals. Safe to call on small clouds:
        if fewer than (orient_k + 1) points exist, orientation step is skipped.
        """
        radius = self.cfg.normal_estimation.radius
        max_nn = self.cfg.normal_estimation.max_nn
        orient_k = self.cfg.normal_estimation.orient_k

        if len(pcd.points) == 0:
            return

        # Uses a Hybrid Search (K-nearest neighbors within a radius) to find local surface patches
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )

        # Orient only if enough neighbours exist
        if len(pcd.points) > orient_k:
            pcd.orient_normals_consistent_tangent_plane(k=orient_k)
