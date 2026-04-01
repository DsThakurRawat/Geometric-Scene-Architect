"""
src package — 3D Room Semantic Segmentation
Lazy-imports heavy GUI modules (interactive_viewer) to avoid crashing
headless / CI environments that lack a display.
"""
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
