import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import logging
from typing import List, Optional, Union

from src.semantic_labeler import LABEL_COLORS
from src.models import PlaneResult, ClusterResult, OutputConfig

logger = logging.getLogger(__name__)


class TopDownMapper:
    """
    Module 8: 2D Top-Down Map Generation
    Projects segmented clusters onto the XY plane and generates a floor plan image.
    """

    _MAP_COLORS = {**LABEL_COLORS, "wall": [0.25, 0.25, 0.25]}

    def __init__(self, config: Union[OutputConfig, dict]):
        if isinstance(config, dict):
            self.cfg = OutputConfig(**config.get("output", {}))
        else:
            self.cfg = config

    def generate(
        self, planes: List[PlaneResult], clusters: List[ClusterResult],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generates a 2D semantic floor-plan image."""
        if output_path is None:
            output_path = "outputs/topdown_map.png"

        all_xy = []
        for plane in planes:
            inlier_cloud = getattr(plane, "inlier_cloud", None) if not isinstance(plane, dict) else plane.get("inlier_cloud")
            if inlier_cloud:
                pts = np.asarray(inlier_cloud.points)
                if len(pts) > 0:
                    all_xy.append(pts[:, :2])
        for cluster in clusters:
            cloud = getattr(cluster, "cloud", None) if not isinstance(cluster, dict) else cluster.get("cloud")
            if cloud:
                pts = np.asarray(cloud.points)
                if len(pts) > 0:
                    all_xy.append(pts[:, :2])

        if not all_xy:
            logger.warning("TopDownMapper: nothing to draw.")
            return None

        combined_xy = np.vstack(all_xy)
        min_xy = combined_xy.min(axis=0)
        max_xy = combined_xy.max(axis=0)
        margin = 0.5

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#1a1a2e")
        ax.set_xlim(min_xy[0] - margin, max_xy[0] + margin)
        ax.set_ylim(min_xy[1] - margin, max_xy[1] + margin)

        for plane in planes:
            if plane.label == "wall" and plane.inlier_cloud:
                pts = np.asarray(inlier_cloud.points)
                if len(pts) > 0:
                    ax.scatter(pts[:, 0], pts[:, 1], s=1,
                               c=[self._MAP_COLORS["wall"]], alpha=0.6, zorder=2)

        seen_labels = set()
        for i, cluster in enumerate(clusters):
            label = cluster.label
            color = self._MAP_COLORS.get(label, self._MAP_COLORS.get("unknown", [0.6, 0, 0.6]))
            bbox = cluster.aabb_box
            if bbox is None:
                continue

            min_b = np.asarray(bbox.min_bound)
            max_b = np.asarray(bbox.max_bound)
            width = float(max_b[0] - min_b[0])
            depth = float(max_b[1] - min_b[1])

            legend_label = label if label not in seen_labels else "_nolegend_"
            seen_labels.add(label)

            rect = patches.Rectangle(
                (float(min_b[0]), float(min_b[1])), width, depth,
                linewidth=1.5, edgecolor="white", facecolor=color,
                alpha=0.75, label=legend_label, zorder=3,
            )
            ax.add_patch(rect)
            cx, cy = cluster.centroid[0], cluster.centroid[1]
            ax.text(cx, cy, f"{label}\n({i})", fontsize=7, ha="center", va="center",
                    color="white", weight="bold", zorder=4)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_title("2D Top-Down Semantic Map", color="white", fontsize=13, weight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize="small", facecolor="#2d2d2d", labelcolor="white")

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        logger.info(f"Top-down map saved to: {output_path}")
        return output_path
