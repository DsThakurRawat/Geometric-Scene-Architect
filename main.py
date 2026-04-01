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


def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    # ── 0. Load Configuration ─────────────────────────────────────────────
    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)
    cfg = load_config(args.config)
    print("--- 3D Room Segmentation Pipeline Started ---")
    print(f"Input : {args.input}")
    print(f"Config: {args.config}")

    # ── 1. Module 1: Load Data ─────────────────────────────────────────────
    loader = PointCloudLoader()
    pcd_raw = loader.load(args.input)
    stats = loader.validate(pcd_raw)
    print(f"Loaded {stats['n_points']:,} points | "
          f"Scene dims: {[f'{d:.2f}m' for d in stats['scene_dims']]}")

    # Normalize: floor -> Z=0
    pcd_norm = loader.normalize_orientation(pcd_raw)

    # ── 2. Module 2: Preprocessing ────────────────────────────────────────
    prep = Preprocessor(cfg)
    pcd_down = prep.voxel_downsample(pcd_norm)
    pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
    prep.estimate_normals(pcd_clean)
    print(f"After preprocessing: {len(pcd_clean.points):,} points remaining.")

    # ── 3. Module 3: Structural Segmentation (RANSAC) ────────────────────
    ransac = IterativeRANSAC(cfg)
    planes, residual_pcd = ransac.extract_planes(pcd_clean)
    print(f"Found {len(planes)} planar structures. "
          f"Residual: {len(residual_pcd.points):,} points.")

    # ── 4. Module 4: Residual Clustering (DBSCAN) ────────────────────────
    clusterer = DBSCANClusterer(cfg)
    clusters = clusterer.cluster(residual_pcd)
    print(f"Found {len(clusters)} object-like clusters.")

    # ── 5. Module 5: Semantic Labeling ───────────────────────────────────
    labeler = SemanticLabeler(cfg)
    stats_clean = loader.validate(pcd_clean)
    scene_height = stats_clean["scene_dims"][2]

    labeled_planes = labeler.label_planes(planes, scene_height=scene_height)
    labeled_clusters = labeler.label_clusters(clusters)

    # ── 6. Module 6: Bounding Box Estimation ─────────────────────────────
    bbox_est = BoundingBoxEstimator()
    labeled_clusters = bbox_est.compute(labeled_clusters)

    # ── 7. Module 8: Export ───────────────────────────────────────────────
    exporter = Exporter(cfg)
    if not labeled_planes and not labeled_clusters:
        print("Warning: no planes or clusters found — skipping PLY export.")
        output_ply = None
        output_json = exporter.export_report(labeled_planes, labeled_clusters)
    else:
        output_ply = exporter.merge_and_export_ply(labeled_planes, labeled_clusters)
        output_json = exporter.export_report(labeled_planes, labeled_clusters)

    print("--- Execution Summary ---")
    print(f"  Planes found    : {len(labeled_planes)}")
    print(f"  Clusters found  : {len(labeled_clusters)}")
    print(f"  Segmented PLY   : {output_ply}")
    print(f"  Report JSON     : {output_json}")

    # ── 8. Module 7: Top-Down Map ─────────────────────────────────────────
    if not args.no_topdown:
        mapper = TopDownMapper(cfg)
        map_path = mapper.generate(labeled_planes, labeled_clusters)
        print(f"  Top-down map    : {map_path}")

    # ── 9. Optional: Screenshot ───────────────────────────────────────────
    if args.screenshot:
        viz = Visualizer()
        screenshot_path = cfg.get("output", {}).get("screenshot", "outputs/screenshot.png")
        viz.save_screenshot(labeled_planes, labeled_clusters, screenshot_path)
        print(f"  Screenshot      : {screenshot_path}")

    print("--- Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()
