"""Tests for IterativeRANSAC — 20 test cases."""
import pytest
import numpy as np
import open3d as o3d
from src.ransac_extractor import IterativeRANSAC


def _flat_floor(n=2000, z_noise=0.005, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    x = rng.uniform(0, 5, n)
    y = rng.uniform(0, 6, n)
    z = rng.normal(0.0, z_noise, n)
    pts = np.column_stack([x, y, z]).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _two_plane_scene():
    """Floor at Z=0 + ceiling at Z=3."""
    rng = np.random.default_rng(11)
    floor_pts = np.column_stack([rng.uniform(0, 5, 1500), rng.uniform(0, 6, 1500), rng.normal(0, 0.005, 1500)])
    ceil_pts  = np.column_stack([rng.uniform(0, 5, 1500), rng.uniform(0, 6, 1500), rng.normal(3, 0.005, 1500)])
    pts = np.vstack([floor_pts, ceil_pts]).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ── extract_planes ────────────────────────────────────────────────────────────

class TestIterativeRANSAC:
    def test_finds_at_least_one_plane(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        assert len(planes) >= 1

    def test_empty_cloud_returns_empty(self, base_cfg):
        empty = o3d.geometry.PointCloud()
        extractor = IterativeRANSAC(base_cfg)
        planes, residual = extractor.extract_planes(empty)
        assert planes == []
        assert len(residual.points) == 0

    def test_required_keys_in_result(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        for p in planes:
            assert "plane_model" in p
            assert "inlier_cloud" in p
            assert "normal" in p
            assert "centroid_z" in p
            assert "inlier_count" in p

    def test_residual_smaller_than_input(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        _, residual = extractor.extract_planes(flat_floor_pcd)
        assert len(residual.points) < len(flat_floor_pcd.points)

    def test_plane_normal_is_list_of_three(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        for p in planes:
            assert len(p["normal"]) == 3

    def test_inlier_count_positive(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        for p in planes:
            assert p["inlier_count"] > 0

    def test_floor_centroid_z_near_zero(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        # The dominant plane (floor) should have centroid_z near 0
        assert abs(planes[0]["centroid_z"]) < 0.5

    def test_two_planes_found_in_two_plane_scene(self, base_cfg):
        scene = _two_plane_scene()
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(scene)
        assert len(planes) >= 2

    def test_max_planes_limits_output(self, base_cfg, flat_floor_pcd):
        base_cfg["ransac"]["max_planes"] = 1
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        assert len(planes) <= 1

    def test_min_plane_size_filters_small_planes(self, base_cfg, flat_floor_pcd):
        base_cfg["ransac"]["min_plane_size"] = 999999
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        assert planes == []  # no plane meets the size threshold

    def test_plane_model_is_four_elements(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        for p in planes:
            assert len(p["plane_model"]) == 4

    def test_floor_normal_z_component_dominant(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        # For a horizontal floor, |nz| should dominate
        n = np.array(planes[0]["normal"])
        assert abs(n[2]) > abs(n[0]) and abs(n[2]) > abs(n[1])

    def test_remaining_points_min_causes_early_stop(self, base_cfg, flat_floor_pcd):
        base_cfg["ransac"]["remaining_points_min"] = 999999
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        # Stops instantly — may return 0 or 1 plane
        assert len(planes) <= 1

    def test_inlier_cloud_is_point_cloud(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        planes, _ = extractor.extract_planes(flat_floor_pcd)
        for p in planes:
            assert isinstance(p["inlier_cloud"], o3d.geometry.PointCloud)

    def test_residual_is_point_cloud(self, base_cfg, flat_floor_pcd):
        extractor = IterativeRANSAC(base_cfg)
        _, residual = extractor.extract_planes(flat_floor_pcd)
        assert isinstance(residual, o3d.geometry.PointCloud)
