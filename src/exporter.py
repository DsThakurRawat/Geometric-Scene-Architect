import open3d as o3d
import json
import os
import copy
from typing import Dict, List, Optional


class Exporter:
    """
    Module 8: Exporting Results
    Saves the labeled point cloud as .ply and generates a JSON metadata report.
    """

    def __init__(self, config: Dict):
        self.cfg = config.get("output", {})

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _safe_makedirs(path: str) -> None:
        """Creates parent directories; handles the case where path has no directory."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ── PLY export ───────────────────────────────────────────────────────

    def merge_and_export_ply(
        self, planes: List[Dict], clusters: List[Dict], output_path: Optional[str] = None
    ) -> str:
        """
        Deep-copies every labeled cloud, merges them, and writes a single colored .ply.
        Deep-copy prevents mutating the caller's clouds when merge is applied.
        """
        if output_path is None:
            output_path = self.cfg.get("ply", "outputs/segmented_room.ply")

        all_pcds = []
        for plane in planes:
            all_pcds.append(copy.deepcopy(plane["inlier_cloud"]))
        for cluster in clusters:
            all_pcds.append(copy.deepcopy(cluster["cloud"]))

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
        planes: List[Dict],
        clusters: List[Dict],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Writes a JSON report with per-plane and per-object metadata.
        Includes OBB dimensions when BoundingBoxEstimator has been run.
        """
        if output_path is None:
            output_path = self.cfg.get("report", "outputs/segmentation_report.json")

        report: Dict = {
            "structural_planes": [],
            "objects": [],
        }

        for i, plane in enumerate(planes):
            report["structural_planes"].append(
                {
                    "plane_id": i,
                    "label": plane.get("label", "unknown"),
                    "inlier_count": plane.get("inlier_count", 0),
                    "centroid_z": plane.get("centroid_z", 0.0),
                    "normal": [round(v, 5) for v in plane.get("normal", [0, 0, 0])],
                    "plane_model": [round(v, 5) for v in plane.get("plane_model", [0, 0, 0, 0])],
                }
            )

        for i, cluster in enumerate(clusters):
            obj: Dict = {
                "cluster_id": i,
                "label": cluster.get("label", "unknown"),
                "n_points": cluster.get("n_points", 0),
                "dims": [round(v, 4) for v in cluster.get("dims", [0, 0, 0])],
                "centroid": [round(v, 4) for v in cluster.get("centroid", [0, 0, 0])],
                "z_min": round(cluster.get("z_min", 0.0), 4),
                "z_max": round(cluster.get("z_max", 0.0), 4),
                "footprint_m2": round(cluster.get("footprint_m2", 0.0), 4),
            }
            # Include OBB info if available
            if "obb_extent" in cluster:
                obj["obb_extent"] = [round(v, 4) for v in cluster["obb_extent"]]
                obj["obb_rotation_deg"] = round(cluster.get("obb_rotation_deg", 0.0), 2)
            report["objects"].append(obj)

        self._safe_makedirs(output_path)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Exported report: {output_path}")

        # ── Optional: Runtime Pydantic Validation ──────────────────────────
        from src.models import SegmentationReport
        try:
            SegmentationReport(**report)
            print("  [Validation] JSON report satisfies Pydantic schema.")
        except Exception as e:
            print(f"  [Validation Warning] Exported report does not strictly match schema: {e}")

        return output_path
