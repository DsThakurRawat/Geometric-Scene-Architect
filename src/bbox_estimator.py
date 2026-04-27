import numpy as np
import logging
from typing import List, Union
from src.models import PlaneResult, ClusterResult

logger = logging.getLogger(__name__)


class BoundingBoxEstimator:
    """
    Module 6: Bounding Boxes & Furniture Dimensions
    Computes AABB and OBB for each cluster.
    
    INTERVIEW TIP: AABB vs OBB?
    AABB (Axis-Aligned): Fixed to the world X, Y, Z axes. Fast but loose for rotated objects.
    OBB (Oriented): Rotates to fit the object's orientation. Much more accurate for calculating 'true' width/depth.
    """

    def compute(self, clusters: List[ClusterResult]) -> List[ClusterResult]:
        """Annotates each cluster with AABB and OBB geometry and summaries."""
        # Loop through every cluster identified by the clustering module.
        for cluster in clusters:
            # Check if the cluster is a Pydantic model or a dictionary (for flexibility).
            # Retrieve the Open3D PointCloud object stored in the 'cloud' field.
            cloud = getattr(cluster, 'cloud', None) if not isinstance(cluster, dict) else cluster.get('cloud')
            
            # If the cloud is missing or has no points, we can't calculate a bounding box.
            if cloud is None or len(cloud.points) == 0:
                cluster_id = getattr(cluster, 'cluster_id', 'unknown') if not isinstance(cluster, dict) else cluster.get('cluster_id', 'unknown')
                logger.warning(f"  [Skipping] Cluster {cluster_id} has 0 points.")
                continue

            # ── AXIS-ALIGNED BOUNDING BOX (AABB) ──────────────────────────
            # AABB is the smallest box that contains all points, aligned with the world X, Y, Z axes.
            aabb = cloud.get_axis_aligned_bounding_box()
            # Set the box color to Red for visualization.
            aabb.color = (1.0, 0.0, 0.0)
            # Calculate dimensions by subtracting the min coordinate from the max coordinate.
            dims = np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)
            
            # Store the AABB object and dimensions back into the cluster record.
            if not isinstance(cluster, dict):
                cluster.aabb_box = aabb
                cluster.dims = dims.tolist()
            else:
                cluster["aabb_box"] = aabb
                cluster["dims"] = dims.tolist()

            # ── ORIENTED BOUNDING BOX (OBB) ──────────────────────────────
            # OBB rotates to find the tightest fit for the object, regardless of world axes.
            # Initialize with default values (AABB dimensions and 0 rotation).
            obb_box, obb_extent, obb_rotation_deg = None, dims.tolist(), 0.0
            try:
                # Open3D needs at least 4 points to compute a valid OBB.
                if len(cloud.points) >= 4:
                    # Get the OBB object from Open3D.
                    obb = cloud.get_oriented_bounding_box()
                    # Set the box color to Green for visualization.
                    obb.color = (0.0, 1.0, 0.0)
                    # Extract the rotation matrix from the OBB.
                    R = np.asarray(obb.R)
                    # Calculate the 'yaw' (rotation around Z axis) in degrees using the rotation matrix.
                    yaw_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
                    # Update variables with actual OBB data.
                    obb_box, obb_extent, obb_rotation_deg = obb, obb.extent.tolist(), yaw_deg
            except Exception:
                # If OBB calculation fails (e.g. collinear points), we fall back to AABB defaults.
                pass

            # Store the OBB object, its extent (true dimensions), and rotation back into the cluster record.
            if not isinstance(cluster, dict):
                cluster.obb_box = obb_box
                cluster.obb_extent = obb_extent
                cluster.obb_rotation_deg = obb_rotation_deg
            else:
                cluster["obb_box"] = obb_box
                cluster["obb_extent"] = obb_extent
                cluster["obb_rotation_deg"] = obb_rotation_deg

            # Log the final results for this object.
            label = getattr(cluster, 'label', 'unknown') if not isinstance(cluster, dict) else cluster.get('label', 'unknown')
            cluster_id = getattr(cluster, 'cluster_id', 'unknown') if not isinstance(cluster, dict) else cluster.get('cluster_id', 'unknown')
            logger.info(
                f"  [{label}] id={cluster_id} "
                f"W={dims[0]:.2f}m D={dims[1]:.2f}m H={dims[2]:.2f}m "
                f"yaw={obb_rotation_deg:.1f}°"
            )

        # Return the updated list of clusters.
        return clusters
