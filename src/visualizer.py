import open3d as o3d
import copy
import os
import logging
from typing import List
from src.models import PlaneResult, ClusterResult
from src.semantic_labeler import LABEL_COLORS

logger = logging.getLogger(__name__)


class Visualizer:
    """Module 8: Visualization — renders segmented point clouds."""

    def get_geometries(self, planes: List[PlaneResult], clusters: List[ClusterResult]) -> List:
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
            if cluster.aabb_box is not None:
                geometries.append(cluster.aabb_box)
            if cluster.obb_box is not None:
                geometries.append(cluster.obb_box)

        return geometries

    def show(self, planes: List[PlaneResult], clusters: List[ClusterResult]) -> None:
        geometries = self.get_geometries(planes, clusters)
        if not geometries:
            logger.warning("Visualizer: nothing to show.")
            return
        o3d.visualization.draw_geometries(geometries, window_name="Semantic Segmentation Result", zoom=0.5)

    def save_screenshot(
        self, planes: List[PlaneResult], clusters: List[ClusterResult], output_path: str
    ) -> None:
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        geometries = self.get_geometries(planes, clusters)
        if not geometries:
            logger.warning("Visualizer: nothing to screenshot.")
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
            logger.info(f"Screenshot saved to: {output_path}")
        except Exception as e:
            logger.warning(f"Could not save screenshot ({e}). Try Xvfb on headless servers.")
