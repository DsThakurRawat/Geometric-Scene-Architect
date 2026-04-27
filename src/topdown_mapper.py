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
        # If no output path is provided, use the default from the configuration.
        if output_path is None:
            output_path = getattr(self.cfg, 'screenshot', "outputs/topdown_map.png")

        # Gather all XY coordinates to determine the overall boundaries (extent) of the map.
        all_xy = []
        # Extract XY points from every detected plane.
        for plane in planes:
            inlier_cloud = getattr(plane, "inlier_cloud", None) if not isinstance(plane, dict) else plane.get("inlier_cloud")
            if inlier_cloud:
                pts = np.asarray(inlier_cloud.points)
                if len(pts) > 0:
                    # We only care about X and Y for a top-down view.
                    all_xy.append(pts[:, :2])
        # Extract XY points from every detected object cluster.
        for cluster in clusters:
            cloud = getattr(cluster, "cloud", None) if not isinstance(cluster, dict) else cluster.get("cloud")
            if cloud:
                pts = np.asarray(cloud.points)
                if len(pts) > 0:
                    all_xy.append(pts[:, :2])

        # If no points were found, we can't draw anything.
        if not all_xy:
            logger.warning("TopDownMapper: nothing to draw.")
            return None

        # Combine all gathered points into a single large array.
        combined_xy = np.vstack(all_xy)
        # Find the minimum and maximum X and Y values to set the plot limits.
        min_xy = combined_xy.min(axis=0)
        max_xy = combined_xy.max(axis=0)
        # Add a small margin (0.5 metres) so the drawing doesn't touch the edge of the image.
        margin = 0.5

        # Create a new Matplotlib figure and axis.
        fig, ax = plt.subplots(figsize=(10, 8))
        # Set a dark background color for a premium "blueprint" look.
        ax.set_facecolor("#1a1a2e")
        # Set the axis limits based on the calculated boundaries and margin.
        ax.set_xlim(min_xy[0] - margin, max_xy[0] + margin)
        ax.set_ylim(min_xy[1] - margin, max_xy[1] + margin)

        # Draw the walls as a background layer.
        for plane in planes:
            label = getattr(plane, "label", "unknown") if not isinstance(plane, dict) else plane.get("label", "unknown")
            inlier_cloud = getattr(plane, "inlier_cloud", None) if not isinstance(plane, dict) else plane.get("inlier_cloud")
            # Only draw planes labeled as walls.
            if label == "wall" and inlier_cloud:
                pts = np.asarray(inlier_cloud.points)
                if len(pts) > 0:
                    # Use a scatter plot with very small points to represent the wall cross-section.
                    ax.scatter(pts[:, 0], pts[:, 1], s=1,
                               c=[self._MAP_COLORS["wall"]], alpha=0.6, zorder=2)

        # Track which labels we've seen to avoid duplicating legend entries.
        seen_labels = set()
        # Iterate through every object cluster to draw its footprint.
        for i, cluster in enumerate(clusters):
            # Determine the label and the associated color.
            label = getattr(cluster, 'label', 'unknown') if not isinstance(cluster, dict) else cluster.get('label', 'unknown')
            color = self._MAP_COLORS.get(label, self._MAP_COLORS.get("unknown", [0.6, 0, 0.6]))
            # Use the Axis-Aligned Bounding Box (AABB) for the footprint.
            bbox = getattr(cluster, 'aabb_box', None) if not isinstance(cluster, dict) else cluster.get('aabb_box', cluster.get('aabb'))
            if bbox is None:
                continue

            # Extract box corners to calculate width and depth.
            min_b = np.asarray(bbox.min_bound)
            max_b = np.asarray(bbox.max_bound)
            width = float(max_b[0] - min_b[0])
            depth = float(max_b[1] - min_b[1])

            # Manage legend labeling.
            legend_label = label if label not in seen_labels else "_nolegend_"
            seen_labels.add(label)

            # Create a rectangle patch representing the object's footprint.
            rect = patches.Rectangle(
                (float(min_b[0]), float(min_b[1])), width, depth,
                linewidth=1.5, edgecolor="white", facecolor=color,
                alpha=0.75, label=legend_label, zorder=3,
            )
            # Add the rectangle to the axis.
            ax.add_patch(rect)
            
            # Place a text label in the center of the object.
            centroid = getattr(cluster, 'centroid', [0,0,0]) if not isinstance(cluster, dict) else cluster.get('centroid', [0,0,0])
            cx, cy = centroid[0], centroid[1]
            ax.text(cx, cy, f"{label}\n({i})", fontsize=7, ha="center", va="center",
                    color="white", weight="bold", zorder=4)

        # Ensure the X and Y scales are equal (1m horizontal = 1m vertical).
        ax.set_aspect("equal")
        # Set labels and title with white text to contrast with the dark background.
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_title("2D Top-Down Semantic Map", color="white", fontsize=13, weight="bold")
        # Set ticks and spine colors to white.
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        # If we have labels, add a legend.
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
