"""
End-to-end integration test for the full pipeline — 10 test cases.
Uses the synthetic_room_pcd fixture from conftest.py.
"""
import json
import os
import pytest
import open3d as o3d

from src.loader import PointCloudLoader
from src.preprocessor import Preprocessor
from src.ransac_extractor import IterativeRANSAC
from src.dbscan_clusterer import DBSCANClusterer
from src.semantic_labeler import SemanticLabeler
from src.bbox_estimator import BoundingBoxEstimator
from src.topdown_mapper import TopDownMapper
from src.exporter import Exporter


def _run_pipeline(pcd_path, cfg, tmp_path):
    """Helper: full pipeline from PLY file to outputs."""
    loader = PointCloudLoader()
    pcd = loader.load(pcd_path)
    pcd = loader.normalize_orientation(pcd)

    prep = Preprocessor(cfg)
    pcd_down = prep.voxel_downsample(pcd)
    pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
    prep.estimate_normals(pcd_clean)

    ransac = IterativeRANSAC(cfg)
    planes, residual = ransac.extract_planes(pcd_clean)

    clusterer = DBSCANClusterer(cfg)
    clusters = clusterer.cluster(residual)

    labeler = SemanticLabeler(cfg)
    scene_height = loader.validate(pcd_clean)["scene_dims"][2]
    planes = labeler.label_planes(planes, scene_height=scene_height)
    clusters = labeler.label_clusters(clusters)

    est = BoundingBoxEstimator()
    clusters = est.compute(clusters)

    ply_path = str(tmp_path / "segmented.ply")
    report_path = str(tmp_path / "report.json")
    map_path = str(tmp_path / "map.png")

    cfg["output"]["ply"] = ply_path
    cfg["output"]["report"] = report_path
    cfg["output"]["screenshot"] = map_path

    exporter = Exporter(cfg)
    exporter.merge_and_export_ply(planes, clusters)
    exporter.export_report(planes, clusters)

    mapper = TopDownMapper(cfg)
    mapper.generate(planes, clusters)

    return planes, clusters, ply_path, report_path, map_path


class TestFullPipeline:
    def test_pipeline_runs_without_error(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        _run_pipeline(path, base_cfg, tmp_path)

    def test_at_least_one_plane_found(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        planes, _, *_ = _run_pipeline(path, base_cfg, tmp_path)
        assert len(planes) >= 1

    def test_all_planes_have_labels(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        planes, _, *_ = _run_pipeline(path, base_cfg, tmp_path)
        for p in planes:
            assert "label" in p and p["label"]

    def test_all_clusters_have_labels(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        _, clusters, *_ = _run_pipeline(path, base_cfg, tmp_path)
        for c in clusters:
            assert "label" in c and c["label"]

    def test_floor_detected(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        planes, _, *_ = _run_pipeline(path, base_cfg, tmp_path)
        labels = [p["label"] for p in planes]
        assert "floor" in labels

    def test_segmented_ply_created_and_loadable(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        _, _, ply_path, *_ = _run_pipeline(path, base_cfg, tmp_path)
        assert os.path.exists(ply_path)
        pcd = o3d.io.read_point_cloud(ply_path)
        assert len(pcd.points) > 0

    def test_json_report_valid(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        _, _, _, report_path, _ = _run_pipeline(path, base_cfg, tmp_path)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            data = json.load(f)
        assert "structural_planes" in data
        assert "objects" in data

    def test_json_plane_count_matches_runtime(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        planes, _, _, report_path, _ = _run_pipeline(path, base_cfg, tmp_path)
        with open(report_path) as f:
            data = json.load(f)
        assert len(data["structural_planes"]) == len(planes)

    def test_topdown_map_saved(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        *_, map_path = _run_pipeline(path, base_cfg, tmp_path)
        assert os.path.exists(map_path)

    def test_clusters_have_bbox_after_estimator(self, synthetic_room_pcd, base_cfg, tmp_path):
        path, _ = synthetic_room_pcd
        _, clusters, *_ = _run_pipeline(path, base_cfg, tmp_path)
        for c in clusters:
            assert "aabb_box" in c
            assert "obb_extent" in c
