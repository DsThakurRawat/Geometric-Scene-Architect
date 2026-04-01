import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe in headless / test environments
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from typing import Dict, List, Optional

from src.semantic_labeler import LABEL_COLORS


class TopDownMapper:
    """
    Module 7: 2D Top-Down Map Generation (Extra Credit)
    Projects segmented clusters onto the XY plane and generates a floor plan image.
    """

    # Slightly darker wall color for 2D contrast
    _MAP_COLORS = {**LABEL_COLORS, "wall": [0.25, 0.25, 0.25]}

    def __init__(self, config: Dict):
        self.cfg = config.get("output", {})

    def generate(
        self,
        planes: List[Dict],
        clusters: List[Dict],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates a 2D semantic floor-plan image.

        Returns the output path on success, or None if there is nothing to draw.
        """
        if output_path is None:
            output_path = self.cfg.get("screenshot", "outputs/topdown_map.png")

        # ── Collect all 2D (XY) points for axis bounds ────────────────────
        # Use ALL planes for bounds (not just floor) so wall-only inputs work.
        all_xy: List[np.ndarray] = []
        for plane in planes:
            pts = np.asarray(plane["inlier_cloud"].points)
            if len(pts) > 0:
                all_xy.append(pts[:, :2])
        for cluster in clusters:
            pts = np.asarray(cluster["cloud"].points)
            if len(pts) > 0:
                all_xy.append(pts[:, :2])

        if not all_xy:
            print("TopDownMapper: nothing to draw (no planes or clusters).")
            return None

        combined_xy = np.vstack(all_xy)
        min_xy = combined_xy.min(axis=0)
        max_xy = combined_xy.max(axis=0)
        margin = 0.5

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_facecolor("#1a1a2e")  # dark background
        ax.set_xlim(min_xy[0] - margin, max_xy[0] + margin)
        ax.set_ylim(min_xy[1] - margin, max_xy[1] + margin)

        # ── Draw wall footprints ──────────────────────────────────────────
        for plane in planes:
            if plane.get("label") == "wall":
                pts = np.asarray(plane["inlier_cloud"].points)
                if len(pts) > 0:
                    ax.scatter(
                        pts[:, 0], pts[:, 1],
                        s=1, c=[self._MAP_COLORS["wall"]], alpha=0.6, zorder=2
                    )

        # ── Draw cluster footprints ───────────────────────────────────────
        seen_labels: set = set()
        for i, cluster in enumerate(clusters):
            label = cluster.get("label", "unknown")
            color = self._MAP_COLORS.get(label, self._MAP_COLORS.get("unknown", [0.6, 0, 0.6]))

            # Use aabb_box if BBoxEstimator ran, fall back to aabb
            bbox = cluster.get("aabb_box") or cluster.get("aabb")
            if bbox is None:
                continue

            min_b = np.asarray(bbox.min_bound)
            max_b = np.asarray(bbox.max_bound)
            width = float(max_b[0] - min_b[0])
            depth = float(max_b[1] - min_b[1])

            legend_label = label if label not in seen_labels else "_nolegend_"
            seen_labels.add(label)

            rect = patches.Rectangle(
                (float(min_b[0]), float(min_b[1])),
                width, depth,
                linewidth=1.5,
                edgecolor="white",
                facecolor=color,
                alpha=0.75,
                label=legend_label,
                zorder=3,
            )
            ax.add_patch(rect)

            cx, cy = cluster["centroid"][0], cluster["centroid"][1]
            ax.text(
                cx, cy, f"{label}\n({i})",
                fontsize=7, ha="center", va="center",
                color="white", weight="bold", zorder=4
            )

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_title("2D Top-Down Semantic Map", color="white", fontsize=13, weight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")

        # Only render legend when there are labelled artists (avoids matplotlib UserWarning)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize="small", facecolor="#2d2d2d", labelcolor="white")

        # ── Save ─────────────────────────────────────────────────────────
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        print(f"Top-down map saved to: {output_path}")
        return output_path
