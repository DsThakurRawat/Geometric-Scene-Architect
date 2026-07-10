"""
src package — 3D Room Semantic Segmentation
Lazy-imports heavy GUI modules (interactive_viewer) to avoid crashing
headless / CI environments that lack a display.
"""
import os as _os
# Reproducibility: Open3D's RANSAC (segment_plane) is OpenMP-parallel and NOT deterministic
# across threads even with a fixed seed — at inlier ties different threads can win, so plane
# membership (and thus the locked benchmark numbers) can wobble by a few points run-to-run.
# Pin Open3D to a single thread for byte-reproducible geometry. This runs before the imports
# below pull in Open3D, so it takes effect for anything that imports `src`. Set OMP_NUM_THREADS
# in the environment to override (speed over exact reproducibility).
_os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.loader import PointCloudLoader
from src.preprocessor import Preprocessor
from src.ransac_extractor import IterativeRANSAC
from src.dbscan_clusterer import DBSCANClusterer
from src.semantic_labeler import SemanticLabeler, LABEL_COLORS
from src.bbox_estimator import BoundingBoxEstimator
from src.topdown_mapper import TopDownMapper
from src.visualizer import Visualizer
from src.exporter import Exporter
from src.models import PlaneResult, ClusterResult, SegmentationReport

__all__ = [
    "PointCloudLoader",
    "Preprocessor",
    "IterativeRANSAC",
    "DBSCANClusterer",
    "SemanticLabeler",
    "LABEL_COLORS",
    "BoundingBoxEstimator",
    "TopDownMapper",
    "Visualizer",
    "Exporter",
    "PlaneResult",
    "ClusterResult",
    "SegmentationReport",
]
