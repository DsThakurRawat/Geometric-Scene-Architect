"""Tests for Preprocessor — 20 test cases."""
import pytest
import numpy as np
import open3d as o3d
from src.preprocessor import Preprocessor


def _make_pcd(n=500, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    pts = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _make_pcd_with_outliers(n_core=500, n_outliers=50):
    rng = np.random.default_rng(3)
    core = rng.uniform(0, 1, (n_core, 3))
    outliers = rng.uniform(100, 200, (n_outliers, 3))
    pts = np.vstack([core, outliers]).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


# ── voxel_downsample ─────────────────────────────────────────────────────────

class TestVoxelDownsample:
    def test_reduces_point_count(self, base_cfg):
        pcd = _make_pcd(2000)
        prep = Preprocessor(base_cfg)
        down = prep.voxel_downsample(pcd)
        assert len(down.points) < len(pcd.points)

    def test_large_voxel_removes_more(self, base_cfg):
        pcd = _make_pcd(2000)
        base_cfg["preprocessing"]["voxel_size"] = 0.01
        prep_fine = Preprocessor(base_cfg)
        fine = prep_fine.voxel_downsample(pcd)

        base_cfg["preprocessing"]["voxel_size"] = 0.5
        prep_coarse = Preprocessor(base_cfg)
        coarse = prep_coarse.voxel_downsample(pcd)

        assert len(coarse.points) < len(fine.points)

    def test_output_is_point_cloud(self, base_cfg):
        pcd = _make_pcd(200)
        prep = Preprocessor(base_cfg)
        result = prep.voxel_downsample(pcd)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_single_point_cloud_downsampled(self, base_cfg):
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        prep = Preprocessor(base_cfg)
        result = prep.voxel_downsample(pcd)
        assert len(result.points) >= 1

    def test_invalid_voxel_size_raises(self, base_cfg):
        base_cfg["preprocessing"]["voxel_size"] = -0.1
        pcd = _make_pcd(200)
        prep = Preprocessor(base_cfg)
        with pytest.raises(ValueError):
            prep.voxel_downsample(pcd)

    def test_default_voxel_size_used_when_missing(self):
        pcd = _make_pcd(2000)
        prep = Preprocessor({})  # empty config
        result = prep.voxel_downsample(pcd)
        assert isinstance(result, o3d.geometry.PointCloud)


# ── remove_statistical_outliers ───────────────────────────────────────────────

class TestStatisticalOutlierRemoval:
    def test_removes_distant_outliers(self, base_cfg):
        pcd = _make_pcd_with_outliers(500, 50)
        prep = Preprocessor(base_cfg)
        clean, outliers = prep.remove_statistical_outliers(pcd)
        assert len(clean.points) < len(pcd.points)
        assert len(outliers.points) > 0

    def test_clean_plus_outliers_equals_original(self, base_cfg):
        pcd = _make_pcd_with_outliers(500, 50)
        prep = Preprocessor(base_cfg)
        clean, outliers = prep.remove_statistical_outliers(pcd)
        assert len(clean.points) + len(outliers.points) == len(pcd.points)

    def test_too_small_for_sor_returns_original(self, base_cfg):
        """If n_points <= nb_neighbors, SOR is skipped."""
        pcd = _make_pcd(5)  # fewer than nb_neighbors=10
        prep = Preprocessor(base_cfg)
        clean, outliers = prep.remove_statistical_outliers(pcd)
        assert len(clean.points) == 5
        assert len(outliers.points) == 0

    def test_clean_cloud_is_pcd(self, base_cfg):
        pcd = _make_pcd(200)
        prep = Preprocessor(base_cfg)
        clean, _ = prep.remove_statistical_outliers(pcd)
        assert isinstance(clean, o3d.geometry.PointCloud)

    def test_strict_ratio_removes_more(self, base_cfg):
        pcd = _make_pcd_with_outliers(500, 100)
        base_cfg["preprocessing"]["sor"]["std_ratio"] = 0.5
        prep_strict = Preprocessor(base_cfg)
        clean_strict, _ = prep_strict.remove_statistical_outliers(pcd)

        base_cfg["preprocessing"]["sor"]["std_ratio"] = 4.0
        prep_lenient = Preprocessor(base_cfg)
        clean_lenient, _ = prep_lenient.remove_statistical_outliers(pcd)

        assert len(clean_strict.points) <= len(clean_lenient.points)


# ── estimate_normals ─────────────────────────────────────────────────────────

class TestEstimateNormals:
    def test_normals_assigned(self, base_cfg):
        pcd = _make_pcd(300)
        prep = Preprocessor(base_cfg)
        assert not pcd.has_normals()
        prep.estimate_normals(pcd)
        assert pcd.has_normals()

    def test_normals_are_unit_length(self, base_cfg):
        pcd = _make_pcd(300)
        prep = Preprocessor(base_cfg)
        prep.estimate_normals(pcd)
        norms = np.linalg.norm(np.asarray(pcd.normals), axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_empty_pcd_does_not_crash(self, base_cfg):
        empty = o3d.geometry.PointCloud()
        prep = Preprocessor(base_cfg)
        prep.estimate_normals(empty)  # Must not raise

    def test_small_pcd_skips_orientation(self, base_cfg):
        """Clouds with fewer points than orient_k should not raise."""
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        prep = Preprocessor(base_cfg)
        prep.estimate_normals(pcd)  # Should not crash

    def test_normal_count_matches_points(self, base_cfg):
        pcd = _make_pcd(300)
        prep = Preprocessor(base_cfg)
        prep.estimate_normals(pcd)
        assert len(pcd.normals) == len(pcd.points)
