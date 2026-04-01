"""Tests for Exporter — 15 test cases."""
import json
import os
import pytest
import numpy as np
import open3d as o3d
from src.exporter import Exporter
from tests.conftest import make_plane, make_cluster


def _labeled_plane(centroid_z=0.0):
    p = make_plane([0, 0, 1], centroid_z)
    p["label"] = "floor"
    return p


def _labeled_cluster(z0=0.0, z1=0.75):
    c = make_cluster(z0, z1, [1.0, 1.0, z1 - z0])
    c["label"] = "furniture"
    return c


class TestExporter:
    # ── PLY export ────────────────────────────────────────────────────────

    def test_ply_file_created(self, tmp_path, base_cfg):
        base_cfg["output"]["ply"] = str(tmp_path / "out.ply")
        exporter = Exporter(base_cfg)
        path = exporter.merge_and_export_ply([_labeled_plane()], [_labeled_cluster()])
        assert os.path.exists(path)

    def test_ply_has_correct_extension(self, tmp_path, base_cfg):
        base_cfg["output"]["ply"] = str(tmp_path / "seg.ply")
        exporter = Exporter(base_cfg)
        path = exporter.merge_and_export_ply([_labeled_plane()], [_labeled_cluster()])
        assert path.endswith(".ply")

    def test_ply_is_loadable(self, tmp_path, base_cfg):
        base_cfg["output"]["ply"] = str(tmp_path / "out.ply")
        exporter = Exporter(base_cfg)
        path = exporter.merge_and_export_ply([_labeled_plane()], [_labeled_cluster()])
        pcd = o3d.io.read_point_cloud(path)
        assert len(pcd.points) > 0

    def test_ply_point_count_is_sum(self, tmp_path, base_cfg):
        plane = _labeled_plane()
        cluster = _labeled_cluster()
        base_cfg["output"]["ply"] = str(tmp_path / "out.ply")
        exporter = Exporter(base_cfg)
        path = exporter.merge_and_export_ply([plane], [cluster])
        pcd = o3d.io.read_point_cloud(path)
        expected = len(plane["inlier_cloud"].points) + len(cluster["cloud"].points)
        assert len(pcd.points) == expected

    def test_merge_does_not_mutate_sources(self, tmp_path, base_cfg):
        plane = _labeled_plane()
        cluster = _labeled_cluster()
        n_plane_before = len(plane["inlier_cloud"].points)
        n_cluster_before = len(cluster["cloud"].points)
        base_cfg["output"]["ply"] = str(tmp_path / "out.ply")
        exporter = Exporter(base_cfg)
        exporter.merge_and_export_ply([plane], [cluster])
        assert len(plane["inlier_cloud"].points) == n_plane_before
        assert len(cluster["cloud"].points) == n_cluster_before

    def test_empty_inputs_raise_value_error(self, tmp_path, base_cfg):
        base_cfg["output"]["ply"] = str(tmp_path / "out.ply")
        exporter = Exporter(base_cfg)
        with pytest.raises(ValueError):
            exporter.merge_and_export_ply([], [])

    def test_output_path_parent_created(self, tmp_path, base_cfg):
        deep = str(tmp_path / "a" / "b" / "c" / "out.ply")
        base_cfg["output"]["ply"] = deep
        exporter = Exporter(base_cfg)
        path = exporter.merge_and_export_ply([_labeled_plane()], [])
        assert os.path.exists(path)

    # ── JSON report ───────────────────────────────────────────────────────

    def test_report_file_created(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "report.json")
        exporter = Exporter(base_cfg)
        path = exporter.export_report([_labeled_plane()], [_labeled_cluster()])
        assert os.path.exists(path)

    def test_report_is_valid_json(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "report.json")
        exporter = Exporter(base_cfg)
        path = exporter.export_report([_labeled_plane()], [_labeled_cluster()])
        with open(path) as f:
            data = json.load(f)
        assert "structural_planes" in data
        assert "objects" in data

    def test_report_plane_count_correct(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "report.json")
        exporter = Exporter(base_cfg)
        planes = [_labeled_plane(), _labeled_plane(2.8)]
        path = exporter.export_report(planes, [])
        with open(path) as f:
            data = json.load(f)
        assert len(data["structural_planes"]) == 2

    def test_report_object_count_correct(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "report.json")
        exporter = Exporter(base_cfg)
        clusters = [_labeled_cluster(), _labeled_cluster(0.5, 1.1)]
        path = exporter.export_report([], clusters)
        with open(path) as f:
            data = json.load(f)
        assert len(data["objects"]) == 2

    def test_report_has_normal_field(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "report.json")
        exporter = Exporter(base_cfg)
        path = exporter.export_report([_labeled_plane()], [])
        with open(path) as f:
            data = json.load(f)
        assert "normal" in data["structural_planes"][0]

    def test_report_object_has_footprint(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "report.json")
        exporter = Exporter(base_cfg)
        path = exporter.export_report([], [_labeled_cluster()])
        with open(path) as f:
            data = json.load(f)
        assert "footprint_m2" in data["objects"][0]

    def test_report_parent_dirs_created(self, tmp_path, base_cfg):
        deep = str(tmp_path / "x" / "y" / "report.json")
        base_cfg["output"]["report"] = deep
        exporter = Exporter(base_cfg)
        path = exporter.export_report([], [])
        assert os.path.exists(path)

    def test_empty_report_no_crash(self, tmp_path, base_cfg):
        base_cfg["output"]["report"] = str(tmp_path / "empty_report.json")
        exporter = Exporter(base_cfg)
        path = exporter.export_report([], [])
        with open(path) as f:
            data = json.load(f)
        assert data["structural_planes"] == []
        assert data["objects"] == []
