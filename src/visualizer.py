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
        """Helper to collect all Open3D objects (clouds, boxes) for display."""
        geometries = []
        # Process every detected plane.
        for plane in planes:
            # If the plane has a point cloud associated with it.
            if plane.inlier_cloud:
                # Create a copy so we don't accidentally modify the original data.
                pcd = copy.deepcopy(plane.inlier_cloud)
                # Determine the correct color based on the semantic label.
                color = LABEL_COLORS.get(plane.label, LABEL_COLORS["unknown"])
                # Apply the color to the cloud.
                pcd.paint_uniform_color(color)
                # Add the cloud to our list of items to draw.
                geometries.append(pcd)

        # Process every detected object cluster.
        for cluster in clusters:
            # If the cluster has a point cloud.
            if cluster.cloud:
                # Create a copy to prevent accidental side effects.
                pcd = copy.deepcopy(cluster.cloud)
                # Lookup the canonical color for this furniture/object label.
                color = LABEL_COLORS.get(cluster.label, LABEL_COLORS["unknown"])
                # Apply the color.
                pcd.paint_uniform_color(color)
                # Add the cloud to our list.
                geometries.append(pcd)
            # If the cluster has an Axis-Aligned Bounding Box, add it to the visualization.
            if cluster.aabb_box is not None:
                geometries.append(cluster.aabb_box)
            # If the cluster has an Oriented Bounding Box, add it too.
            if cluster.obb_box is not None:
                geometries.append(cluster.obb_box)

        # Return the final list of all geometries.
        return geometries

    def show(self, planes: List[PlaneResult], clusters: List[ClusterResult]) -> None:
        """Opens an interactive 3D window to view the results."""
        # Collect all visual objects.
        geometries = self.get_geometries(planes, clusters)
        # If there's nothing to draw, log a warning and exit.
        if not geometries:
            logger.warning("Visualizer: nothing to show.")
            return
        # Open the Open3D interactive viewer.
        o3d.visualization.draw_geometries(geometries, window_name="Semantic Segmentation Result", zoom=0.5)

    def save_screenshot(
        self, planes: List[PlaneResult], clusters: List[ClusterResult], output_path: str
    ) -> None:
        """Saves a high-resolution rendering of the scene to a file (headless-friendly)."""
        # Ensure the output directory exists.
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Gather all geometries.
        geometries = self.get_geometries(planes, clusters)
        # Exit if nothing to draw.
        if not geometries:
            logger.warning("Visualizer: nothing to screenshot.")
            return
        try:
            # Create a non-visible visualization window.
            vis = o3d.visualization.Visualizer()
            # Set resolution (e.g. 1080p).
            vis.create_window(visible=False, width=1920, height=1080)
            # Add all items to the scene.
            for geom in geometries:
                vis.add_geometry(geom)
            # Run the rendering engine briefly to process the objects.
            vis.poll_events()
            vis.update_renderer()
            # Save the frame as an image.
            vis.capture_screen_image(output_path)
            # Close the window to free resources.
            vis.destroy_window()
            logger.info(f"Screenshot saved to: {output_path}")
        except Exception as e:
            # Fallback if no display server is available (common on servers).
            logger.warning(f"Could not save screenshot ({e}). Try Xvfb on headless servers.")
