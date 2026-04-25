"""
main.py — 3D Room Scene Semantic Segmentation Pipeline

Usage:
    python main.py --input data/synthetic/room_01.ply --config configs/default.yaml
    python main.py --input data/processed/office_1.ply --config configs/default.yaml
"""
import yaml
import argparse
import os
import sys

from src.loader import PointCloudLoader
from src.preprocessor import Preprocessor
from src.ransac_extractor import IterativeRANSAC
from src.dbscan_clusterer import DBSCANClusterer
from src.semantic_labeler import SemanticLabeler
from src.bbox_estimator import BoundingBoxEstimator
from src.topdown_mapper import TopDownMapper
from src.visualizer import Visualizer
from src.exporter import Exporter
from src.models import PipelineConfig


def load_config(config_path: str) -> PipelineConfig:
    """Loads a YAML configuration file and validates it via Pydantic."""
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return PipelineConfig(**raw_cfg)


def main():
    parser = argparse.ArgumentParser(description="3D Room Scene Semantic Segmentation Pipeline")
    parser.add_argument("--input", type=str, default="data/synthetic/room_01.ply",
                        help="Input point cloud file (.ply or .pcd).")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="YAML configuration file.")
    parser.add_argument("--no-topdown", action="store_true",
                        help="Skip 2D top-down map generation.")
    parser.add_argument("--screenshot", action="store_true",
                        help="Save a headless rendering screenshot.")
    args = parser.parse_args()

    # ── 0. Load & Validate Configuration ──────────────────────────────────
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    
    print("--- 3D Room Segmentation Pipeline Started ---")

    # ── 1. Module 1: Load Data ─────────────────────────────────────────────
    loader = PointCloudLoader()
    pcd_raw = loader.load(args.input)
    stats = loader.validate(pcd_raw)
    print(f"Loaded {stats['n_points']:,} points.")

    # Normalize: floor -> Z=0
    pcd_norm = loader.normalize_orientation(pcd_raw)

    # ── 2. Module 2: Preprocessing ────────────────────────────────────────
    prep = Preprocessor(cfg.preprocessing)
    pcd_down = prep.voxel_downsample(pcd_norm)
    pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
    prep.estimate_normals(pcd_clean)
    print(f"After preprocessing: {len(pcd_clean.points):,} points remaining.")

    # ── 3. Module 3: Structural Segmentation (RANSAC) ────────────────────
    ransac = IterativeRANSAC(cfg.ransac)
    planes, residual_pcd = ransac.extract_planes(pcd_clean)
    print(f"Found {len(planes)} planar structures.")

    # ── 4. Module 4: Residual Clustering (DBSCAN) ────────────────────────
    clusterer = DBSCANClusterer(cfg.dbscan)
    clusters = clusterer.cluster(residual_pcd)
    print(f"Found {len(clusters)} object-like clusters.")

    # ── 5. Module 5: Semantic Labeling ───────────────────────────────────
    labeler = SemanticLabeler(cfg.labeling)
    stats_clean = loader.validate(pcd_clean)
    scene_height = stats_clean["scene_dims"][2]

    labeled_planes = labeler.label_planes(planes, scene_height=scene_height)
    labeled_clusters = labeler.label_clusters(clusters)

    # ── 6. Module 6: Bounding Box Estimation ─────────────────────────────
    bbox_est = BoundingBoxEstimator()
    labeled_clusters = bbox_est.compute(labeled_clusters)

    # ── 7. Module 7: Export ───────────────────────────────────────────────
    exporter = Exporter(cfg.output)
    output_ply = exporter.merge_and_export_ply(labeled_planes, labeled_clusters)
    output_json = exporter.export_report(labeled_planes, labeled_clusters)

    # ── 8. Module 8: Top-Down Map ─────────────────────────────────────────
    if not args.no_topdown:
        mapper = TopDownMapper(cfg.output)
        mapper.generate(labeled_planes, labeled_clusters)

    # ── 9. Optional: Screenshot ───────────────────────────────────────────
    if args.screenshot:
        viz = Visualizer()
        viz.save_screenshot(labeled_planes, labeled_clusters, cfg.output.screenshot)

    print(f"--- Pipeline Finished Successfully ---")
    print(f"Output PLY : {output_ply}")
    print(f"Output JSON: {output_json}")


if __name__ == "__main__":
    main()
