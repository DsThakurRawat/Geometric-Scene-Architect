import open3d as o3d
import copy
import os
from typing import List
from src.models import PlaneResult, ClusterResult
from src.semantic_labeler import LABEL_COLORS


class Visualizer:
    """
    Module 8: Visualization
    Renders segmented point clouds and saves headless screenshots.
    """

    def get_geometries(self, planes: List[PlaneResult], clusters: List[ClusterResult]) -> List:
        """Returns a list of colored geometry objects for rendering."""
        geometries = []
        for plane in planes:
            if plane.inlier_cloud:
                pcd = copy.deepcopy(plane.inlier_cloud)
                color = LABEL_COLORS.get(plane.label, LABEL_COLORS["unknown"])
                pcd.paint_uniform_color(color)
                geometries.append(pcd)

        for cluster in clusters:
            if cluster.cloud:
                pcd = copy.deepcopy(cluster.cloud)
                color = LABEL_COLORS.get(cluster.label, LABEL_COLORS["unknown"])
                pcd.paint_uniform_color(color)
                geometries.append(pcd)

            # Include bounding boxes if BoundingBoxEstimator has been run
            if cluster.aabb_box is not None:
                geometries.append(cluster.aabb_box)
            if cluster.obb_box is not None:
                geometries.append(cluster.obb_box)

        return geometries

    def show(self, planes: List[PlaneResult], clusters: List[ClusterResult]) -> None:
        """Opens an interactive 3D viewer. Closes when the user presses Q."""
        geometries = self.get_geometries(planes, clusters)
        if not geometries:
            print("Visualizer: nothing to show.")
            return
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Semantic Segmentation Result",
            zoom=0.5,
        )

    def save_screenshot(
        self,
        planes: List[PlaneResult],
        clusters: List[ClusterResult],
        output_path: str,
    ) -> None:
        """
        Saves a headless screenshot of the segmented scene.
        Requires a display (X server or virtual framebuffer).
        Falls back gracefully if the display is unavailable.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        geometries = self.get_geometries(planes, clusters)
        if not geometries:
            print("Visualizer: nothing to screenshot.")
            return

        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False, width=1920, height=1080)
            for geom in geometries:
                vis.add_geometry(geom)
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(output_path)
            vis.destroy_window()
            print(f"Screenshot saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save screenshot ({e}). "
                  "Try running with a virtual display (Xvfb) on headless servers.")
