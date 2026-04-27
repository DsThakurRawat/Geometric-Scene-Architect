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
import logging
import time
import traceback
from typing import Dict

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

# ── Logging Configuration ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 0. Load & Validate Configuration ──────────────────────────────────
    try:
        if not os.path.exists(args.config):
            logger.error(f"Config file not found at {args.config}")
            sys.exit(1)
        cfg = load_config(args.config)
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    logger.info("--- 3D Room Segmentation Pipeline Started ---")
    # INTERVIEW TIP: Pipeline Overview
    # The pipeline follows a 'Cascading Extraction' pattern:
    # 1. Load & Clean (Module 1-2)
    # 2. Extract Dominant Geometry (Module 3 - RANSAC)
    # 3. Cluster Residuals (Module 4 - DBSCAN)
    # 4. Classify & Measure (Module 5-6)
    # 5. Export (Module 7-8)
    logger.info(f"Input : {args.input}")
    logger.info(f"Config: {args.config}")

    timings: Dict[str, float] = {}
    total_start = time.perf_counter()

    try:
        # ── Module 1: Load & Normalize ────────────────────────────────────
        # PURPOSE: Reading raw data and ensuring the floor is aligned with Z=0.
        start = time.perf_counter()
        loader = PointCloudLoader()
        pcd_raw = loader.load(args.input)
        stats = loader.validate(pcd_raw)
        logger.info(f"Loaded {stats['n_points']:,} points | "
                    f"Scene dims: {[f'{d:.2f}m' for d in stats['scene_dims']]}")
        pcd_norm = loader.normalize_orientation(pcd_raw)
        timings["Load & Normalize"] = time.perf_counter() - start

        # ── Module 2: Preprocessing ───────────────────────────────────────
        # PURPOSE: Denoising and downsampling to speed up processing.
        # Estimating normals is CRITICAL for RANSAC to detect planes accurately.
        start = time.perf_counter()
        prep = Preprocessor(cfg.preprocessing)
        pcd_down = prep.voxel_downsample(pcd_norm)
        pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
        prep.estimate_normals(pcd_clean)
        logger.info(f"After preprocessing: {len(pcd_clean.points):,} points remaining.")
        timings["Preprocessing"] = time.perf_counter() - start

        # ── Module 3: Structural Extraction (RANSAC) ─────────────────────
        # PURPOSE: Finding flat surfaces (walls, floors).
        # We remove these points so they don't interfere with object clustering.
        start = time.perf_counter()
        ransac = IterativeRANSAC(cfg.ransac)
        planes, residual_pcd = ransac.extract_planes(pcd_clean)
        logger.info(f"Found {len(planes)} planar structures. "
                    f"Residual: {len(residual_pcd.points):,} points.")
        timings["RANSAC"] = time.perf_counter() - start

        # ── Module 4: Residual Clustering (DBSCAN) ───────────────────────
        # PURPOSE: Grouping the non-planar points into individual objects (chairs, tables).
        start = time.perf_counter()
        clusterer = DBSCANClusterer(cfg.dbscan)
        clusters = clusterer.cluster(residual_pcd)
        logger.info(f"Found {len(clusters)} object-like clusters.")
        timings["DBSCAN"] = time.perf_counter() - start

        # ── Module 5: Semantic Labeling ──────────────────────────────────
        # PURPOSE: Using geometric heuristics to decide what each plane/cluster is.
        start = time.perf_counter()
        labeler = SemanticLabeler(cfg.labeling)
        stats_clean = loader.validate(pcd_clean)
        scene_height = stats_clean["scene_dims"][2]
        labeled_planes = labeler.label_planes(planes, scene_height=scene_height)
        labeled_clusters = labeler.label_clusters(clusters)
        timings["Labeling"] = time.perf_counter() - start

        # ── Module 6: Bounding Box Estimation ────────────────────────────
        # PURPOSE: Calculating the tightest possible box (OBB) around each object
        # to get true physical dimensions (width, depth, height).
        start = time.perf_counter()
        # Initialize the estimator.
        bbox_est = BoundingBoxEstimator()
        # Run the calculation for all clusters.
        labeled_clusters = bbox_est.compute(labeled_clusters)
        # Store execution time.
        timings["BBox Estimation"] = time.perf_counter() - start

        # ── Module 7: Exporting Results ──────────────────────────────────
        # PURPOSE: Saving the segmented 3D cloud and the summary report.
        start = time.perf_counter()
        # Initialize the exporter.
        exporter = Exporter(cfg.output)
        # If nothing was found, log a warning.
        if not labeled_planes and not labeled_clusters:
            logger.warning("No planes or clusters found — skipping PLY export.")
            output_ply = None
            # Still export the (empty) report for consistency.
            output_json = exporter.export_report(labeled_planes, labeled_clusters)
        else:
            # Save the merged, colored point cloud.
            output_ply = exporter.merge_and_export_ply(labeled_planes, labeled_clusters)
            # Save the JSON report.
            output_json = exporter.export_report(labeled_planes, labeled_clusters)
        timings["Export"] = time.perf_counter() - start

        # ── Module 8: Top-Down Map Generation ────────────────────────────
        # PURPOSE: Creating a 2D floor-plan view of the segmented scene.
        if not args.no_topdown:
            start = time.perf_counter()
            # Initialize the mapper.
            mapper = TopDownMapper(cfg.output)
            # Generate the image file.
            map_path = mapper.generate(labeled_planes, labeled_clusters)
            logger.info(f"Top-down map: {map_path}")
            timings["Top-Down Map"] = time.perf_counter() - start

        # ── Optional: Screenshot ─────────────────────────────────────────
        # If requested, save a high-res static image of the 3D result.
        if args.screenshot:
            viz = Visualizer()
            viz.save_screenshot(labeled_planes, labeled_clusters, cfg.output.screenshot)
            logger.info(f"Screenshot saved.")

        # Log final summary statistics to the console.
        total_time = time.perf_counter() - total_start
        logger.info("--- Execution Summary ---")
        logger.info(f"  Planes found    : {len(labeled_planes)}")
        logger.info(f"  Clusters found  : {len(labeled_clusters)}")
        logger.info(f"  Segmented PLY   : {output_ply}")
        logger.info(f"  Report JSON     : {output_json}")

        # Log detailed timing breakdown for each stage.
        logger.info("--- Timing Summary ---")
        for stage, duration in timings.items():
            logger.info(f"  {stage:<18}: {duration:.3f}s")
        logger.info(f"  {'Total':<18}: {total_time:.3f}s")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    logger.info("--- Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()
