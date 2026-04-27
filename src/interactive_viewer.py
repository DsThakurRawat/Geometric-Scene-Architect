"""
Module 9: Interactive Viewer (Extra Credit)
Open3D GUI-based semantic labeling tool.

Features:
- Left panel: cluster list with auto-detected labels
- Viewport: 3D scene view
- Right panel: label dropdown + Apply button
- Bottom: export button to save corrected labels
"""
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import json
import os
from typing import List, Dict

from src.semantic_labeler import LABEL_COLORS

LABEL_OPTIONS = ["floor", "ceiling", "wall", "furniture", "tall_furniture",
                 "small_object", "high_fixture", "unknown"]


class SemanticViewer:
    """
    Open3D GUI viewer for interactive semantic label correction.
    Allows the user to select clusters, change labels, and export the result.
    """

    def __init__(self, planes: List[Dict], clusters: List[Dict]):
        self.planes = planes
        self.clusters = clusters
        self.selected_idx = None

        self.app = gui.Application.instance
        self.app.initialize()

        self.window = self.app.create_window("Semantic Segmentation Viewer", 1600, 900)
        em = self.window.theme.font_size

        # ── Scene / 3D Widget ──────────────────────────────────────────────
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([0.1, 0.1, 0.1, 1.0])

        # ── Left panel — cluster list ──────────────────────────────────────
        self.left_panel = gui.ScrollableVert(0.25 * em, gui.Margins(0.5 * em))
        self.cluster_buttons = []

        for i, cluster in enumerate(clusters):
            label = cluster.get("label", "unknown")
            btn = gui.Button(f"Obj {i}: {label}")
            btn.set_on_clicked(self._make_select_handler(i))
            self.cluster_buttons.append(btn)
            self.left_panel.add_child(btn)

        # ── Right panel — label selector + apply ──────────────────────────
        self.right_panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em))
        self.right_panel.add_child(gui.Label("Selected Cluster Label"))

        self.label_combo = gui.Combobox()
        for lbl in LABEL_OPTIONS:
            self.label_combo.add_item(lbl)
        self.right_panel.add_child(self.label_combo)

        apply_btn = gui.Button("Apply Label")
        apply_btn.set_on_clicked(self._on_apply_label)
        self.right_panel.add_child(apply_btn)

        self.right_panel.add_child(gui.Label(""))  # spacer

        export_btn = gui.Button("Export Labels to JSON")
        export_btn.set_on_clicked(self._on_export)
        self.right_panel.add_child(export_btn)

        # ── Layout ────────────────────────────────────────────────────────
        layout = gui.Horiz(0, gui.Margins(0))
        left_wrap = gui.Vert(0, gui.Margins(0))
        left_wrap.add_child(self.left_panel)
        left_wrap.set_preferred_width(200)

        right_wrap = gui.Vert(0, gui.Margins(0))
        right_wrap.add_child(self.right_panel)
        right_wrap.set_preferred_width(200)

        layout.add_child(left_wrap)
        layout.add_stretch()
        layout.add_child(self.scene)
        layout.add_stretch()
        layout.add_child(right_wrap)

        self.window.add_child(layout)

        # Load geometries
        self._load_scene()

    def _make_select_handler(self, idx: int):
        def handler():
            self._select_cluster(idx)
        return handler

    def _select_cluster(self, idx: int):
        self.selected_idx = idx
        cluster = self.clusters[idx]
        label = cluster.get("label", "unknown")
        if label in LABEL_OPTIONS:
            self.label_combo.selected_index = LABEL_OPTIONS.index(label)
        print(f"Selected cluster {idx}: {label}")

    def _on_apply_label(self):
        if self.selected_idx is None:
            return
        new_label = LABEL_OPTIONS[self.label_combo.selected_index]
        self.clusters[self.selected_idx]["label"] = new_label
        self.cluster_buttons[self.selected_idx].text = f"Obj {self.selected_idx}: {new_label}"
        # Repaint
        self.clusters[self.selected_idx]["cloud"].paint_uniform_color(
            LABEL_COLORS.get(new_label, LABEL_COLORS["unknown"])
        )
        self._load_scene()

    def _on_export(self):
        out = []
        for i, cluster in enumerate(self.clusters):
            out.append({"id": i, "label": cluster.get("label", "unknown")})
        os.makedirs("outputs", exist_ok=True)
        path = "outputs/corrected_labels.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=4)
        print(f"Exported corrected labels to {path}")

    def _load_scene(self):
        """Clears the 3D viewport and re-adds all geometries (planes and clusters)."""
        # Remove any existing 3D objects from the scene.
        self.scene.scene.clear_geometry()
        # Define a material with basic lighting (shading).
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"

        # Iterate through every structural plane.
        for i, plane in enumerate(self.planes):
            # Get the point cloud.
            pcd = plane["inlier_cloud"]
            # Get the color for its current label.
            color = LABEL_COLORS.get(plane.get("label", "unknown"), LABEL_COLORS["unknown"])
            # Paint the points.
            pcd.paint_uniform_color(color)
            # Add the plane to the 3D scene widget.
            self.scene.scene.add_geometry(f"plane_{i}", pcd, mat)

        # Iterate through every object cluster.
        for i, cluster in enumerate(self.clusters):
            # Get the point cloud.
            pcd = cluster["cloud"]
            # Get the color for its label.
            color = LABEL_COLORS.get(cluster.get("label", "unknown"), LABEL_COLORS["unknown"])
            # Paint it.
            pcd.paint_uniform_color(color)
            # Add it to the scene.
            self.scene.scene.add_geometry(f"cluster_{i}", pcd, mat)

        # Automatically adjust the camera to fit all objects in the view.
        bounds = self.scene.scene.bounding_box
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def run(self):
        """Starts the GUI application event loop."""
        self.app.run()


if __name__ == "__main__":
    # Quick standalone test with synthetic data
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.loader import PointCloudLoader
    from src.preprocessor import Preprocessor
    from src.ransac_extractor import IterativeRANSAC
    from src.dbscan_clusterer import DBSCANClusterer
    from src.semantic_labeler import SemanticLabeler
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    loader = PointCloudLoader()
    pcd = loader.load("data/synthetic/room_01.ply")
    pcd = loader.normalize_orientation(pcd)

    prep = Preprocessor(cfg)
    pcd_down = prep.voxel_downsample(pcd)
    pcd_clean, _ = prep.remove_statistical_outliers(pcd_down)
    prep.estimate_normals(pcd_clean)

    ransac = IterativeRANSAC(cfg)
    planes, residual = ransac.extract_planes(pcd_clean)

    clusterer = DBSCANClusterer(cfg)
    clusters = clusterer.cluster(residual)

    labeler = SemanticLabeler(cfg)
    stats = loader.validate(pcd_clean)
    planes = labeler.label_planes(planes, scene_height=stats["scene_dims"][2])
    clusters = labeler.label_clusters(clusters)

    viewer = SemanticViewer(planes, clusters)
    viewer.run()
