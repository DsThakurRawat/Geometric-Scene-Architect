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
        """Reduces point density using voxel downsampling."""
        voxel_size = self.cfg.voxel_size
        return pcd.voxel_down_sample(voxel_size=voxel_size)

    def remove_statistical_outliers(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Removes floating scan noise via Statistical Outlier Removal.
        Returns (clean_pcd, outlier_pcd). If the input has too few points
        (< nb_neighbors + 1), returns (pcd, empty_pcd) without filtering.
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
        clean_pcd = pcd.select_by_index(ind)
        outlier_pcd = pcd.select_by_index(ind, invert=True)

        # Guard: if SOR removes everything, return original
        if len(clean_pcd.points) == 0:
            return pcd, outlier_pcd

        return clean_pcd, outlier_pcd

    def remove_radius_outliers(
        self, pcd: o3d.geometry.PointCloud
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """
        Removes isolated points via Radius Outlier Removal.
        A point is removed if fewer than `min_neighbors` points exist within `radius`.
        Returns (clean_pcd, outlier_pcd).
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

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )

        # Orient only if enough neighbours exist
        if len(pcd.points) > orient_k:
            pcd.orient_normals_consistent_tangent_plane(k=orient_k)
