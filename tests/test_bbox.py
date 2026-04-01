"""Tests for BoundingBoxEstimator — 10 test cases."""
import numpy as np
import open3d as o3d
from src.bbox_estimator import BoundingBoxEstimator
from tests.conftest import make_cluster


def _cluster_with_label(z0=0.0, z1=0.75, dims=None, label="furniture"):
    if dims is None:
        dims = [1.0, 1.0, z1 - z0]
    c = make_cluster(z0, z1, dims)
    c["label"] = label
    return c


class TestBoundingBoxEstimator:
    def test_aabb_box_key_set(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        assert "aabb_box" in result[0]

    def test_obb_rotation_deg_key_set(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        assert "obb_rotation_deg" in result[0]

    def test_obb_extent_key_set(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        assert "obb_extent" in result[0]

    def test_dims_updated_from_aabb(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        assert len(result[0]["dims"]) == 3

    def test_dims_positive(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        assert all(d >= 0 for d in result[0]["dims"])

    def test_tiny_cluster_does_not_crash(self):
        """3-point cluster (degenerate) should not raise."""
        est = BoundingBoxEstimator()
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        cluster = {
            "label_id": 0, "cloud": pcd, "n_points": 3,
            "aabb": pcd.get_axis_aligned_bounding_box(),
            "centroid": [0.33, 0.33, 0.0],
            "dims": [1.0, 1.0, 0.0],
            "z_min": 0.0, "z_max": 0.0, "footprint_m2": 1.0, "label": "furniture"
        }
        result = est.compute([cluster])
        assert "obb_extent" in result[0]

    def test_obb_extent_is_three_elements(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        assert len(result[0]["obb_extent"]) == 3

    def test_multiple_clusters_all_processed(self):
        est = BoundingBoxEstimator()
        clusters = [_cluster_with_label(z0=i * 0.1, z1=i * 0.1 + 0.5) for i in range(5)]
        result = est.compute(clusters)
        assert len(result) == 5
        for c in result:
            assert "aabb_box" in c and "obb_extent" in c

    def test_modifies_in_place(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        original_id = id(cluster)
        result = est.compute([cluster])
        assert id(result[0]) == original_id

    def test_aabb_color_is_red(self):
        est = BoundingBoxEstimator()
        cluster = _cluster_with_label()
        result = est.compute([cluster])
        color = result[0]["aabb_box"].color
        assert abs(color[0] - 1.0) < 1e-6  # R=1
        assert abs(color[1] - 0.0) < 1e-6  # G=0
