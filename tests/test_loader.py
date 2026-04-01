"""Tests for PointCloudLoader — 20 test cases."""
import pytest
import numpy as np
import open3d as o3d
from pathlib import Path
from src.loader import PointCloudLoader


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_pcd(tmp_path, pts, name="test.ply"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
    path = str(tmp_path / name)
    o3d.io.write_point_cloud(path, pcd)
    return path


# ── load() ────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_ply_succeeds(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        assert len(pcd.points) == 200

    def test_load_returns_point_cloud(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        assert isinstance(pcd, o3d.geometry.PointCloud)

    def test_load_missing_file_raises(self):
        loader = PointCloudLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/tmp/definitely_does_not_exist_xyz.ply")

    def test_load_unsupported_extension_raises(self, tmp_path):
        path = str(tmp_path / "mesh.obj")
        Path(path).write_text("v 0 0 0\n")
        loader = PointCloudLoader()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(path)

    def test_load_all_supported_extension_strings(self):
        loader = PointCloudLoader()
        for ext in [".ply", ".pcd", ".xyz", ".xyzrgb", ".pts"]:
            assert ext in loader.SUPPORTED_EXTENSIONS

    def test_load_preserves_point_count(self, tmp_path):
        rng = np.random.default_rng(0)
        for n in [100, 500, 2000]:
            path = _write_pcd(tmp_path, rng.uniform(0, 1, (n, 3)), f"n{n}.ply")
            loader = PointCloudLoader()
            pcd = loader.load(path)
            assert len(pcd.points) == n


# ── validate() ────────────────────────────────────────────────────────────────

class TestValidate:
    def test_validate_has_all_keys(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        stats = loader.validate(pcd)
        for key in ["n_points", "has_colors", "has_normals",
                    "bbox_min", "bbox_max", "scene_dims", "centroid"]:
            assert key in stats

    def test_validate_n_points_correct(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        stats = loader.validate(pcd)
        assert stats["n_points"] == 200

    def test_validate_scene_dims_positive(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        stats = loader.validate(pcd)
        assert all(d >= 0 for d in stats["scene_dims"])

    def test_validate_bbox_min_le_max(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        stats = loader.validate(pcd)
        for mn, mx in zip(stats["bbox_min"], stats["bbox_max"]):
            assert mn <= mx

    def test_validate_no_colors(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        assert not loader.validate(pcd)["has_colors"]

    def test_validate_empty_pcd_does_not_crash(self):
        loader = PointCloudLoader()
        empty = o3d.geometry.PointCloud()
        stats = loader.validate(empty)
        assert stats["n_points"] == 0

    def test_validate_colored_pcd(self, tmp_path):
        rng = np.random.default_rng(5)
        pts = rng.uniform(0, 1, (100, 3)).astype(np.float32)
        cols = rng.uniform(0, 1, (100, 3)).astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        path = str(tmp_path / "colored.ply")
        o3d.io.write_point_cloud(path, pcd)
        loader = PointCloudLoader()
        pcd2 = loader.load(path)
        assert loader.validate(pcd2)["has_colors"]


# ── normalize_orientation() ───────────────────────────────────────────────────

class TestNormalizeOrientation:
    def test_floor_near_zero_after_normalize(self, tmp_path):
        rng = np.random.default_rng(10)
        pts = rng.uniform(5, 10, (500, 3)).astype(np.float32)  # all points high up
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        loader = PointCloudLoader()
        loader.normalize_orientation(pcd)
        z_arr = np.asarray(pcd.points)[:, 2]
        assert np.percentile(z_arr, 5) < 0.1

    def test_normalize_empty_pcd_does_not_crash(self):
        loader = PointCloudLoader()
        empty = o3d.geometry.PointCloud()
        result = loader.normalize_orientation(empty)
        assert isinstance(result, o3d.geometry.PointCloud)

    def test_normalize_returns_same_object(self, random_pcd):
        loader = PointCloudLoader()
        pcd = loader.load(random_pcd)
        result = loader.normalize_orientation(pcd)
        assert result is pcd  # in-place

    def test_negative_z_cloud_is_shifted_up(self, tmp_path):
        pts = np.ones((200, 3), dtype=np.float32)
        pts[:, 2] = -5.0  # all at -5
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        loader = PointCloudLoader()
        loader.normalize_orientation(pcd)
        z_arr = np.asarray(pcd.points)[:, 2]
        assert abs(z_arr.mean()) < 0.2
