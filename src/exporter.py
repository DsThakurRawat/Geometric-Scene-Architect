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
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def merge_and_export_ply(
        self, planes: List[PlaneResult], clusters: List[ClusterResult],
        output_path: Optional[str] = None
    ) -> str:
        """Deep-copies every labeled cloud, merges them, and writes a single colored .ply."""
        if output_path is None:
            output_path = self.cfg.ply

        all_pcds = []
        for plane in planes:
            if plane.inlier_cloud:
                all_pcds.append(copy.deepcopy(plane.inlier_cloud))
        for cluster in clusters:
            if cluster.cloud:
                all_pcds.append(copy.deepcopy(cluster.cloud))

        if not all_pcds:
            raise ValueError("No point clouds to export — both planes and clusters are empty.")

        merged = all_pcds[0]
        for pcd in all_pcds[1:]:
            merged += pcd

        self._safe_makedirs(output_path)
        o3d.io.write_point_cloud(output_path, merged)
        logger.info(f"Exported PLY: {output_path}  ({len(merged.points):,} points)")
        return output_path

    def export_report(
        self, planes: List[PlaneResult], clusters: List[ClusterResult],
        output_path: Optional[str] = None,
    ) -> str:
        """Writes a JSON report with per-plane and per-object metadata."""
        if output_path is None:
            output_path = self.cfg.report

        report_model = SegmentationReport(
            structural_planes=planes,
            objects=clusters
        )
        report_dict = report_model.model_dump(mode="json")

        self._safe_makedirs(output_path)
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        logger.info(f"Exported report: {output_path}")
        return output_path
