"""Tests for TopDownMapper — 10 test cases."""
import os
import pytest
import numpy as np
import open3d as o3d
from src.topdown_mapper import TopDownMapper
from tests.conftest import make_plane, make_cluster


def _wall_plane():
    p = make_plane([1, 0, 0], 1.5)
    p["label"] = "wall"
    return p


def _floor_plane():
    p = make_plane([0, 0, 1], 0.01)
    p["label"] = "floor"
    return p


def _furniture_cluster():
    c = make_cluster(0.0, 0.75, [1.0, 1.0, 0.75])
    c["label"] = "furniture"
    return c


class TestTopDownMapper:
    def test_creates_image_file(self, tmp_path, base_cfg):
        base_cfg["output"]["screenshot"] = str(tmp_path / "map.png")
        mapper = TopDownMapper(base_cfg)
        path = mapper.generate([_wall_plane()], [_furniture_cluster()])
        assert os.path.exists(path)

    def test_returns_output_path(self, tmp_path, base_cfg):
        out = str(tmp_path / "map.png")
        base_cfg["output"]["screenshot"] = out
        mapper = TopDownMapper(base_cfg)
        result = mapper.generate([_wall_plane()], [_furniture_cluster()])
        assert result == out

    def test_empty_everything_returns_none(self, tmp_path, base_cfg):
        base_cfg["output"]["screenshot"] = str(tmp_path / "empty.png")
        mapper = TopDownMapper(base_cfg)
        result = mapper.generate([], [])
        assert result is None

    def test_only_clusters_no_planes(self, tmp_path, base_cfg):
        base_cfg["output"]["screenshot"] = str(tmp_path / "only_clusters.png")
        mapper = TopDownMapper(base_cfg)
        result = mapper.generate([], [_furniture_cluster()])
        assert result is not None and os.path.exists(result)

    def test_only_planes_no_clusters(self, tmp_path, base_cfg):
        base_cfg["output"]["screenshot"] = str(tmp_path / "only_planes.png")
        mapper = TopDownMapper(base_cfg)
        result = mapper.generate([_wall_plane()], [])
        assert result is not None and os.path.exists(result)

    def test_nested_output_dir_created(self, tmp_path, base_cfg):
        deep = str(tmp_path / "a" / "b" / "map.png")
        base_cfg["output"]["screenshot"] = deep
        mapper = TopDownMapper(base_cfg)
        mapper.generate([_wall_plane()], [_furniture_cluster()])
        assert os.path.exists(deep)

    def test_uses_aabb_key_when_aabb_box_absent(self, tmp_path, base_cfg):
        """Cluster has only 'aabb' key (before BBoxEstimator ran) — should not crash."""
        base_cfg["output"]["screenshot"] = str(tmp_path / "map.png")
        mapper = TopDownMapper(base_cfg)
        cluster = _furniture_cluster()
        assert "aabb_box" not in cluster  # not set by conftest
        result = mapper.generate([], [cluster])
        assert result is not None

    def test_uses_aabb_box_key_when_present(self, tmp_path, base_cfg):
        """After BBoxEstimator, cluster has 'aabb_box' — should use it."""
        base_cfg["output"]["screenshot"] = str(tmp_path / "map.png")
        mapper = TopDownMapper(base_cfg)
        cluster = _furniture_cluster()
        cluster["aabb_box"] = cluster["aabb"]  # simulate BBoxEstimator output
        result = mapper.generate([], [cluster])
        assert result is not None

    def test_multiple_clusters_all_drawn(self, tmp_path, base_cfg):
        base_cfg["output"]["screenshot"] = str(tmp_path / "multi.png")
        mapper = TopDownMapper(base_cfg)
        clusters = [_furniture_cluster() for _ in range(5)]
        for i, c in enumerate(clusters):
            c["label_id"] = i
        result = mapper.generate([], clusters)
        assert result is not None and os.path.exists(result)

    def test_custom_output_path_overrides_config(self, tmp_path, base_cfg):
        base_cfg["output"]["screenshot"] = str(tmp_path / "default.png")
        custom = str(tmp_path / "custom.png")
        mapper = TopDownMapper(base_cfg)
        result = mapper.generate([_wall_plane()], [_furniture_cluster()], output_path=custom)
        assert result == custom
        assert os.path.exists(custom)
