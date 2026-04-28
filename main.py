"""
File-by-file breakdown
main.py — The Orchestrator (Boss file)

This is the entry point. You run python main.py --input room.ply and it:
1. Reads your config YAML
2. Calls each module one by one (load → clean → find planes → cluster → label → bbox → export)
3. Times every step and prints a summary

Think of it as a factory manager — it doesn't do the actual work, it just tells each worker 
what to do and in what order. It also handles errors (try/except) so if something breaks, 
you get a clean error message instead of a crash.
"""

import yaml             # Import YAML library for parsing configuration files
import argparse         # Import argparse for handling command-line arguments
import os               # Import os for interacting with the operating system (e.g., path checks)
import sys              # Import sys for system-level operations (e.g., exiting the script)
import logging          # Import logging for structured output and debugging information
import time             # Import time to measure the execution duration of pipeline stages
import traceback        # Import traceback to capture and print detailed error logs
from typing import Dict # Import Dict from typing to define dictionary types for clarity

# Import local modules from the src directory
from src.loader import PointCloudLoader           # Import the class for loading and normalizing 3D point clouds
from src.preprocessor import Preprocessor         # Import the class for cleaning and downsampling point clouds
from src.ransac_extractor import IterativeRANSAC  # Import RANSAC logic for extracting geometric planes (walls, floors)
from src.dbscan_clusterer import DBSCANClusterer  # Import DBSCAN for grouping remaining points into objects
from src.semantic_labeler import SemanticLabeler  # Import logic to assign labels (e.g., 'chair', 'wall') to clusters
from src.bbox_estimator import BoundingBoxEstimator # Import logic to calculate physical dimensions (bounding boxes)
from src.topdown_mapper import TopDownMapper      # Import logic to generate a 2D floor plan image
from src.visualizer import Visualizer             # Import logic for 3D visualization and screenshots
from src.exporter import Exporter                 # Import logic to save the results to files (PLY, JSON)
from src.models import PipelineConfig             # Import the Pydantic model for configuration validation

# ── Logging Configuration ──────────────────────────────────────────────────
# Configure the global logging format, level, and timestamp style
logging.basicConfig(
    level=logging.INFO,                                     # Set the default log level to INFO
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', # Define the structure of log messages
    datefmt='%H:%M:%S'                                      # Set the timestamp format to Hour:Minute:Second
)
logger = logging.getLogger(__name__) # Create a logger instance for this specific file

def load_config(config_path: str) -> PipelineConfig:
    """Loads a YAML configuration file and validates it via Pydantic."""
    with open(config_path, "r") as f:        # Open the specified YAML file in read mode
        raw_cfg = yaml.safe_load(f)          # Load the YAML content into a Python dictionary
    return PipelineConfig(**raw_cfg)         # Pass the dictionary to the Pydantic model for validation

def main():
    # Initialize the argument parser with a description of the program
    parser = argparse.ArgumentParser(description="3D Room Scene Semantic Segmentation Pipeline")
    
    # Define command-line argument for the input point cloud file
    parser.add_argument("--input", type=str, default="data/synthetic/room_01.ply",
                        help="Input point cloud file (.ply or .pcd).")
    
    # Define command-line argument for the configuration file
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="YAML configuration file.")
    
    # Define a flag to disable 2D top-down map generation
    parser.add_argument("--no-topdown", action="store_true",
                        help="Skip 2D top-down map generation.")
    
    # Define a flag to save a headless rendering screenshot
    parser.add_argument("--screenshot", action="store_true",
                        help="Save a headless rendering screenshot.")
    
    # Define a flag to enable detailed debug logging
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging.")
    
    # Parse the arguments provided by the user via the terminal
    args = parser.parse_args()

    # If the verbose flag is set, change the global log level to DEBUG
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 0. Load & Validate Configuration ──────────────────────────────────
    try:
        # Check if the configuration file exists at the specified path
        if not os.path.exists(args.config):
            logger.error(f"Config file not found at {args.config}") # Log an error if missing
            sys.exit(1)                                             # Exit the program with an error code
        
        # Load and validate the configuration using the helper function
        cfg = load_config(args.config)
    except Exception as e:
        # Catch any errors during config loading (e.g., syntax errors, missing fields)
        logger.error(f"Configuration Error: {e}") # Log the specific error
        sys.exit(1)                                # Exit the program

    # Log the start of the pipeline and the paths being used
    logger.info("--- 3D Room Segmentation Pipeline Started ---")
    logger.info(f"Input : {args.input}")
    logger.info(f"Config: {args.config}")

    timings: Dict[str, float] = {}   # Dictionary to store the duration of each pipeline module
    total_start = time.perf_counter() # Record the start time of the entire process

    try:
        # ── Module 1: Load & Normalize ────────────────────────────────────
        start = time.perf_counter()                # Record start time for this module
        loader = PointCloudLoader()                # Initialize the loader component
        pcd_raw = loader.load(args.input)          # Load the raw point cloud from the input file
        stats = loader.validate(pcd_raw)           # Perform a health check (point count, dimensions)
        
        # Log the number of points and the physical dimensions of the scene
        logger.info(f"Loaded {stats['n_points']:,} points | "
                    f"Scene dims: {[f'{d:.2f}m' for d in stats['scene_dims']]}")
        
        # Align the floor with the Z=0 plane for consistent processing
        pcd_norm = loader.normalize_orientation(pcd_raw)
        timings["Load & Normalize"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 2: Preprocessing ───────────────────────────────────────
        start = time.perf_counter()                # Record start time for this module
        prep = Preprocessor(cfg.preprocessing)     # Initialize the preprocessor with config settings
        pcd_down = prep.voxel_downsample(pcd_norm) # Reduce density while preserving shape to speed up work
        
        # Remove noisy points that are far away from their neighbors
        pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
        
        # Compute surface normals (required for RANSAC to detect flat planes)
        prep.estimate_normals(pcd_clean)
        
        # Log the number of points remaining after cleaning and downsampling
        logger.info(f"After preprocessing: {len(pcd_clean.points):,} points remaining.")
        timings["Preprocessing"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 3: Structural Extraction (RANSAC) ─────────────────────
        start = time.perf_counter()                 # Record start time for this module
        ransac = IterativeRANSAC(cfg.ransac)        # Initialize the RANSAC extractor
        
        # Detect and extract flat surfaces (walls, floor, ceiling)
        planes, residual_pcd = ransac.extract_planes(pcd_clean)
        
        # Log how many planes were found and how many points are left for clustering
        logger.info(f"Found {len(planes)} planar structures. "
                    f"Residual: {len(residual_pcd.points):,} points.")
        timings["RANSAC"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 4: Residual Clustering (DBSCAN) ───────────────────────
        start = time.perf_counter()                 # Record start time for this module
        clusterer = DBSCANClusterer(cfg.dbscan)     # Initialize the DBSCAN clusterer
        
        # Group the non-planar points into individual objects based on proximity
        clusters = clusterer.cluster(residual_pcd)
        
        # Log the number of potential objects identified
        logger.info(f"Found {len(clusters)} object-like clusters.")
        timings["DBSCAN"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 5: Semantic Labeling ──────────────────────────────────
        start = time.perf_counter()                 # Record start time for this module
        labeler = SemanticLabeler(cfg.labeling)     # Initialize the semantic labeler
        stats_clean = loader.validate(pcd_clean)    # Get final scene stats for height-based heuristics
        scene_height = stats_clean["scene_dims"][2] # Extract the total height of the room
        
        # Assign labels to planes (e.g., 'Floor', 'Wall') based on orientation and height
        labeled_planes = labeler.label_planes(planes, scene_height=scene_height)
        
        # Assign labels to clusters (e.g., 'Table', 'Chair') based on size and position
        labeled_clusters = labeler.label_clusters(clusters)
        timings["Labeling"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 6: Bounding Box Estimation ────────────────────────────
        start = time.perf_counter()                 # Record start time for this module
        bbox_est = BoundingBoxEstimator()           # Initialize the bounding box tool
        
        # Calculate the oriented bounding box (OBB) for each object to get dimensions
        labeled_clusters = bbox_est.compute(labeled_clusters)
        timings["BBox Estimation"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 7: Exporting Results ──────────────────────────────────
        start = time.perf_counter()                 # Record start time for this module
        exporter = Exporter(cfg.output)             # Initialize the file exporter
        
        # Check if any geometry was successfully processed
        if not labeled_planes and not labeled_clusters:
            logger.warning("No planes or clusters found — skipping PLY export.") # Log warning if empty
            output_ply = None                                                   # Set output path to None
            
            # Still attempt to export a report (it will just show 0 objects)
            output_json = exporter.export_report(labeled_planes, labeled_clusters)
        else:
            # Color-code the points by label and save as a standard 3D PLY file
            output_ply = exporter.merge_and_export_ply(labeled_planes, labeled_clusters)
            
            # Save a structured JSON report containing all object metadata and dimensions
            output_json = exporter.export_report(labeled_planes, labeled_clusters)
        
        timings["Export"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Module 8: Top-Down Map Generation ────────────────────────────
        # Skip this if the user provided the --no-topdown flag
        if not args.no_topdown:
            start = time.perf_counter()             # Record start time for this module
            mapper = TopDownMapper(cfg.output)      # Initialize the 2D mapping tool
            
            # Create a 2D floor plan image representing the 3D scene
            map_path = mapper.generate(labeled_planes, labeled_clusters)
            logger.info(f"Top-down map: {map_path}") # Log where the image was saved
            timings["Top-Down Map"] = time.perf_counter() - start # Calculate and store elapsed time

        # ── Optional: Screenshot ─────────────────────────────────────────
        # If the --screenshot flag was provided, render and save a static 3D image
        if args.screenshot:
            viz = Visualizer()                                                     # Initialize the visualizer
            viz.save_screenshot(labeled_planes, labeled_clusters, cfg.output.screenshot) # Save the image
            logger.info(f"Screenshot saved.")                                      # Log success

        # ── Summary Logs ─────────────────────────────────────────────────
        total_time = time.perf_counter() - total_start # Calculate total execution time
        logger.info("--- Execution Summary ---")
        logger.info(f"  Planes found    : {len(labeled_planes)}")      # Summary of planes
        logger.info(f"  Clusters found  : {len(labeled_clusters)}")    # Summary of objects
        logger.info(f"  Segmented PLY   : {output_ply}")              # Path to exported cloud
        logger.info(f"  Report JSON     : {output_json}")             # Path to exported report

        # Log a detailed breakdown of how long each module took to run
        logger.info("--- Timing Summary ---")
        for stage, duration in timings.items():
            logger.info(f"  {stage:<18}: {duration:.3f}s") # Print stage name and duration
        logger.info(f"  {'Total':<18}: {total_time:.3f}s")    # Print total duration

    except Exception as e:
        # Catch any runtime errors that occur during pipeline execution
        logger.error(f"Pipeline failed: {e}")      # Log the error message
        logger.debug(traceback.format_exc())       # Log the full stack trace for debugging if verbose
        sys.exit(1)                                # Exit with error code 1

    # Log successful completion of the entire pipeline
    logger.info("--- Pipeline Finished Successfully ---")

# Standard Python idiom: only run main() if this script is executed directly
if __name__ == "__main__":
    main() # Call the main function to start the program
