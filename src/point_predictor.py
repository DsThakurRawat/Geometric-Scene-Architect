# src/point_predictor.py
"""
Wraps the existing geometry pipeline so it emits a label *per point* on the
downsampled cloud, which the S3DIS evaluator then aligns back to full resolution.

`run_geometry` is the shared core (planes + clusters + the downsampled cloud); it is
reused by the hybrid labeler and the feature-ML training script so every arm sees the
*identical* geometry. `predict` is the geometry-only arm used for the Phase-1 gate.
"""
import yaml
import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from typing import Tuple, Dict, Any, List

from src.loader import PointCloudLoader
from src.preprocessor import Preprocessor
from src.ransac_extractor import IterativeRANSAC
from src.dbscan_clusterer import DBSCANClusterer
from src.semantic_labeler import SemanticLabeler
from src.bbox_estimator import BoundingBoxEstimator
from src.models import PipelineConfig

# structural plane labels the geometry arm is allowed to emit directly
_STRUCTURAL = {"floor", "ceiling", "wall"}


class PointPredictor:
    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        s3dis_config_path: str = "configs/s3dis.yaml",
        seed: int = 42,
    ):
        # Open3D's RANSAC (segment_plane) is randomised; without a fixed seed each
        # run_geometry call yields a slightly different segmentation, so results are
        # not reproducible AND the feature-ML / hybrid arms (which each recompute
        # geometry) would not see the SAME planes. Seeding the global Open3D RNG at
        # the start of every run_geometry makes geometry deterministic per room and
        # identical across arms.
        self._seed = seed

        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)

        # Honour the S3DIS voxel size if a project config is present.
        if os.path.exists(s3dis_config_path):
            with open(s3dis_config_path, "r") as f:
                s3dis_cfg = yaml.safe_load(f)
            if s3dis_cfg and "preprocessing" in s3dis_cfg:
                cfg_dict.setdefault("preprocessing", {})
                cfg_dict["preprocessing"]["voxel_size"] = s3dis_cfg["preprocessing"].get(
                    "voxel_size", cfg_dict["preprocessing"].get("voxel_size", 0.03)
                )

        self.cfg = PipelineConfig(**cfg_dict)

        self.loader = PointCloudLoader()
        self.preprocessor = Preprocessor(self.cfg.preprocessing)
        self.ransac = IterativeRANSAC(self.cfg.ransac)
        self.clusterer = DBSCANClusterer(self.cfg.dbscan)
        self.labeler = SemanticLabeler(self.cfg.labeling)
        self.bbox = BoundingBoxEstimator()

    # ── shared geometry core ─────────────────────────────────────────────────
    def run_geometry(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Run preprocess -> RANSAC -> DBSCAN -> label on an (N,3|6) array.

        Returns a dict with the downsampled cloud actually used and the structural /
        object results, so downstream arms all share the same geometry:
            clean_pts      (M,3) downsampled+cleaned coords in ORIGINAL frame
            tree           cKDTree over clean_pts (for point<-plane/cluster mapping)
            labeled_planes list[PlaneResult] with floor/ceiling/wall/... labels
            clusters       list[ClusterResult] with OBB features filled in
            z_floor        the z offset that was removed then restored
        """
        # Deterministic geometry: seed Open3D's global RNG so RANSAC/DBSCAN are
        # reproducible and identical across every arm that recomputes geometry.
        o3d.utility.random.seed(self._seed)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(points[:, :3], dtype=np.float64))
        if points.shape[1] >= 6:
            colors = np.asarray(points[:, 3:6], dtype=np.float64).copy()
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Normalise: put the (robust) floor at z=0 so absolute-height heuristics fire.
        pts_arr = np.asarray(pcd.points)
        z_floor = float(np.percentile(pts_arr[:, 2], 5))
        pcd.translate([0.0, 0.0, -z_floor])

        pcd_down = self.preprocessor.voxel_downsample(pcd)
        pcd_clean, _ = self.preprocessor.remove_statistical_outliers(pcd_down)
        self.preprocessor.estimate_normals(pcd_clean)

        planes, residual_pcd = self.ransac.extract_planes(pcd_clean)
        clusters = self.clusterer.cluster(residual_pcd)

        stats = self.loader.validate(pcd_clean)
        scene_height = stats["scene_dims"][2]
        labeled_planes = self.labeler.label_planes(planes, scene_height=scene_height)
        clusters = self.labeler.label_clusters(clusters)
        clusters = self.bbox.compute(clusters)

        clean_pts = np.asarray(pcd_clean.points) + np.array([0.0, 0.0, z_floor])
        clean_colors = (
            np.asarray(pcd_clean.colors)
            if pcd_clean.has_colors()
            else np.zeros((len(clean_pts), 3))
        )
        tree = cKDTree(clean_pts)

        return {
            "clean_pts": clean_pts,
            "clean_colors": clean_colors,
            "tree": tree,
            "labeled_planes": labeled_planes,
            "clusters": clusters,
            "z_floor": z_floor,
        }

    def _map_indices(self, tree: cKDTree, geo_pts: np.ndarray, z_floor: float) -> np.ndarray:
        """Nearest clean-cloud index for a set of plane/cluster points (in normalised frame)."""
        if len(geo_pts) == 0:
            return np.empty((0,), dtype=int)
        restored = np.asarray(geo_pts) + np.array([0.0, 0.0, z_floor])
        dists, indices = tree.query(restored, k=1)
        return indices[dists < 1e-5]

    def segments(self, geo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten a run_geometry result into uniform segment records, one per plane and
        per cluster, each carrying its indices into ``clean_pts``, an 8-d feature
        vector, kind, and (for planes) the rule-based structural label.

        Shared by the feature-ML training script (majority-vote GT per segment) and
        the feature-ML / hybrid arms (per-point prediction). Guarantees every arm
        sees the identical geometry.
        """
        from src.feature_ml import extract_cluster_features, extract_plane_features

        tree, z_floor = geo["tree"], geo["z_floor"]
        colors = geo.get("clean_colors")
        out: List[Dict[str, Any]] = []

        def mean_rgb(idx):
            if colors is None or len(idx) == 0:
                return np.zeros(3)
            return colors[idx].mean(axis=0)

        for plane in geo["labeled_planes"]:
            if plane.inlier_cloud is None:
                continue
            idx = self._map_indices(tree, np.asarray(plane.inlier_cloud.points), z_floor)
            if len(idx) == 0:
                continue
            out.append({
                "kind": "plane",
                "indices": idx,
                "features": extract_plane_features(plane),
                "mean_rgb": mean_rgb(idx),
                "structural_label": plane.label if plane.label in _STRUCTURAL else None,
            })

        for cluster in geo["clusters"]:
            if cluster.cloud is None:
                continue
            idx = self._map_indices(tree, np.asarray(cluster.cloud.points), z_floor)
            if len(idx) == 0:
                continue
            out.append({
                "kind": "cluster",
                "indices": idx,
                "features": extract_cluster_features(cluster),
                "mean_rgb": mean_rgb(idx),
                "structural_label": None,
            })

        return out

    # ── geometry-only arm (Phase 1 gate) ─────────────────────────────────────
    def predict(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Geometry-only per-point labels for the STRUCT4 arm.

        Returns:
            points_used (M,3) downsampled coords (original frame)
            pred_labels (M,)  strings: 'ceiling'/'floor'/'wall'/'object' (default 'clutter')
        """
        geo = self.run_geometry(points)
        clean_pts, tree = geo["clean_pts"], geo["tree"]
        z_floor = geo["z_floor"]

        pred_labels = np.full(len(clean_pts), "clutter", dtype=object)

        for plane in geo["labeled_planes"]:
            if plane.inlier_cloud is None:
                continue
            idx = self._map_indices(tree, np.asarray(plane.inlier_cloud.points), z_floor)
            label = plane.label if plane.label in _STRUCTURAL else "object"
            pred_labels[idx] = label

        for cluster in geo["clusters"]:
            if cluster.cloud is None:
                continue
            idx = self._map_indices(tree, np.asarray(cluster.cloud.points), z_floor)
            pred_labels[idx] = "object"

        return clean_pts, pred_labels
