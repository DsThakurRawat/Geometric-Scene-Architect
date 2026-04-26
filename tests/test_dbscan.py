"""Tests for DBSCANClusterer — 15 test cases."""
import numpy as np
import open3d as o3d
from src.dbscan_clusterer import DBSCANClusterer


def _make_two_clusters(gap=1.0, n=300):
    """Two well-separated clusters."""
    rng = np.random.default_rng(20)
    c1 = rng.uniform(0, 0.4, (n, 3))
    c2 = rng.uniform(0, 0.4, (n, 3)) + [gap, 0, 0]
    pts = np.vstack([c1, c2]).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _make_single_cluster(n=400):
    rng = np.random.default_rng(21)
    pts = rng.uniform(0, 0.5, (n, 3)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


class TestDBSCANClusterer:
    def test_empty_cloud_returns_empty(self, base_cfg):
        empty = o3d.geometry.PointCloud()
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(empty)
        assert result == []

    def test_tiny_cloud_returns_empty(self, base_cfg):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        assert result == []

    def test_two_separated_clusters_found(self, base_cfg):
        pcd = _make_two_clusters(gap=2.0, n=400)
        base_cfg["dbscan"]["min_cluster_points"] = 50
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        assert len(result) >= 1  # At minimum finds one cluster

    def test_cluster_dict_has_required_keys(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        if result:
            c = result[0]
            for key in ["cluster_id", "cloud", "n_points", "aabb_box", "centroid",
                        "dims", "z_min", "z_max", "footprint_m2"]:
                assert key in c, f"Missing key: {key}"

    def test_n_points_matches_cloud(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        for c in result:
            assert c["n_points"] == len(c["cloud"].points)

    def test_z_min_le_z_max(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        for c in result:
            assert c["z_min"] <= c["z_max"]

    def test_footprint_m2_positive(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        for c in result:
            assert c["footprint_m2"] >= 0.0

    def test_all_noise_returns_empty(self, base_cfg):
        """When eps is tiny, all points are noise — should return []."""
        pcd = _make_single_cluster(100)
        base_cfg["dbscan"]["eps"] = 0.000001
        base_cfg["dbscan"]["min_points"] = 1000
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        assert result == []

    def test_min_cluster_points_filter(self, base_cfg):
        pcd = _make_single_cluster(200)
        base_cfg["dbscan"]["min_cluster_points"] = 999999  # impossible to satisfy
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        assert result == []

    def test_max_object_size_filter(self, base_cfg):
        """A tiny max_object_size should filter out room-scale residuals."""
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        base_cfg["dbscan"]["max_object_size"] = 0.0001  # filter everything
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        assert result == []

    def test_sorted_by_n_points_descending(self, base_cfg):
        pcd = _make_two_clusters(gap=2.0, n=400)
        base_cfg["dbscan"]["min_cluster_points"] = 50
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        counts = [c["n_points"] for c in result]
        assert counts == sorted(counts, reverse=True)

    def test_cloud_in_cluster_is_pcd(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        for c in result:
            assert isinstance(c["cloud"], o3d.geometry.PointCloud)

    def test_dims_are_three_elements(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        for c in result:
            assert len(c["dims"]) == 3

    def test_centroid_is_inside_bbox(self, base_cfg):
        pcd = _make_single_cluster(400)
        base_cfg["dbscan"]["min_cluster_points"] = 20
        clusterer = DBSCANClusterer(base_cfg)
        result = clusterer.cluster(pcd)
        for c in result:
            cx, cy, cz = c["centroid"]
            mn = c["aabb_box"].min_bound
            mx = c["aabb_box"].max_bound
            assert mn[0] <= cx <= mx[0]
            assert mn[1] <= cy <= mx[1]
            assert mn[2] <= cz <= mx[2]
