import open3d as o3d
import json
import os
import copy
import logging
from typing import List, Optional, Union
from src.models import PlaneResult, ClusterResult, SegmentationReport, OutputConfig

logger = logging.getLogger(__name__)


class Exporter:
    """
    Module 7: Exporting Results
    Saves the labeled point cloud as .ply and generates a JSON metadata report.
    """

    def __init__(self, config: Union[OutputConfig, dict]):
        if isinstance(config, dict):
            self.cfg = OutputConfig(**config.get("output", {}))
        else:
            self.cfg = config

    @staticmethod
    def _safe_makedirs(path: str) -> None:
        """Ensures that the directory for a given file path exists."""
        # Get the directory part of the full file path.
        parent = os.path.dirname(path)
        # If there is a directory component and it doesn't exist, create it.
        if parent:
            os.makedirs(parent, exist_ok=True)

    def merge_and_export_ply(
        self, planes: List[PlaneResult], clusters: List[ClusterResult],
        output_path: Optional[str] = None
    ) -> str:
        """Deep-copies every labeled cloud, merges them, and writes a single colored .ply."""
        # Use the provided output path or fall back to the configuration default.
        if output_path is None:
            output_path = getattr(self.cfg, "ply", "outputs/segmented.ply")

        # Create an empty list to hold all the individual point cloud chunks.
        all_pcds = []
        # Add all plane clouds to the list.
        for plane in planes:
            # Safely retrieve the 'inlier_cloud' object.
            cloud = getattr(plane, "inlier_cloud", None) if not isinstance(plane, dict) else plane.get("inlier_cloud")
            if cloud:
                # Copy the cloud so that any downstream operations don't modify the memory of the original.
                all_pcds.append(copy.deepcopy(cloud))
        # Add all object/furniture clouds to the list.
        for cluster in clusters:
            # Safely retrieve the 'cloud' object.
            cloud = getattr(cluster, "cloud", None) if not isinstance(cluster, dict) else cluster.get("cloud")
            if cloud:
                # Copy the cloud.
                all_pcds.append(copy.deepcopy(cloud))

        # If we have no clouds to export, something is wrong with the pipeline.
        if not all_pcds:
            raise ValueError("No point clouds to export — both planes and clusters are empty.")

        # Take the first cloud as our base.
        merged = all_pcds[0]
        # Append every other cloud to the base cloud. 
        # Open3D supports the '+' operator for merging two point clouds.
        for pcd in all_pcds[1:]:
            merged += pcd

        # Create the folder if it doesn't exist.
        self._safe_makedirs(output_path)
        # Save the combined cloud as a binary PLY file.
        o3d.io.write_point_cloud(output_path, merged)
        # Log success and the final point count.
        logger.info(f"Exported PLY: {output_path}  ({len(merged.points):,} points)")
        # Return the path where the file was saved.
        return output_path

    def export_report(
        self, planes: List[PlaneResult], clusters: List[ClusterResult],
        output_path: Optional[str] = None,
    ) -> str:
        """Writes a JSON report with per-plane and per-object metadata."""
        # Determine target file path.
        if output_path is None:
            output_path = getattr(self.cfg, "report", "outputs/report.json")

        # Ensure all data is in the correct Pydantic model format for serialization.
        validated_planes = [PlaneResult(**p) if isinstance(p, dict) else p for p in planes]
        validated_clusters = [ClusterResult(**c) if isinstance(c, dict) else c for c in clusters]

        # Construct the final report container.
        report_model = SegmentationReport(
            structural_planes=validated_planes,
            objects=validated_clusters
        )
        # Convert the models into a dictionary suitable for JSON (handling types like NumPy arrays).
        report_dict = report_model.model_dump(mode="json")

        # Ensure directory existence.
        self._safe_makedirs(output_path)
        # Open the file for writing.
        with open(output_path, "w") as f:
            # Dump the dictionary as a formatted (indented) JSON string.
            json.dump(report_dict, f, indent=4)

        # Log completion.
        logger.info(f"Exported report: {output_path}")
        # Return path.
        return output_path
