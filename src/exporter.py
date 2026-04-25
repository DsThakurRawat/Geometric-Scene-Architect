import open3d as o3d
import json
import os
import copy
from typing import List, Optional, Union
from src.models import PlaneResult, ClusterResult, SegmentationReport, OutputConfig


class Exporter:
    """
    Module 8: Exporting Results
    Saves the labeled point cloud as .ply and generates a JSON metadata report.
    """

    def __init__(self, config: Union[OutputConfig, dict]):
        if isinstance(config, dict):
            self.cfg = OutputConfig(**config.get("output", {}))
        else:
            self.cfg = config

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_makedirs(path: str) -> None:
        """Creates parent directories; handles the case where path has no directory."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ── PLY export ───────────────────────────────────────────────────────

    def merge_and_export_ply(
        self, planes: List[PlaneResult], clusters: List[ClusterResult], output_path: Optional[str] = None
    ) -> str:
        """
        Deep-copies every labeled cloud, merges them, and writes a single colored .ply.
        Deep-copy prevents mutating the caller's clouds when merge is applied.
        """
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
        print(f"Exported PLY: {output_path}  ({len(merged.points):,} points)")
        return output_path

    # ── JSON report ──────────────────────────────────────────────────────

    def export_report(
        self,
        planes: List[PlaneResult],
        clusters: List[ClusterResult],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Writes a JSON report with per-plane and per-object metadata.
        Includes OBB dimensions when BoundingBoxEstimator has been run.
        """
        if output_path is None:
            output_path = self.cfg.report

        # Use Pydantic model for construction to ensure correctness
        report_model = SegmentationReport(
            structural_planes=planes,
            objects=clusters
        )

        # Convert to dict for serialization (this handles the Field(exclude=True) logic)
        report_dict = report_model.model_dump(mode="json")

        self._safe_makedirs(output_path)
        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=4)

        print(f"Exported report: {output_path}")
        return output_path
