"""
conftest.py — shared pytest fixtures for all test modules.

Fixture categories:
    - Basic clouds:     Minimal point clouds for unit-level module tests.
    - Config fixtures:  Pre-built pipeline configuration dicts.
    - Factories:        Helper functions (not fixtures) that create planes/clusters
                        with controllable geometry for parametrised tests.
"""
import pytest
import numpy as np
import open3d as o3d


# ── Basic clouds ─────────────────────────────────────────────────────────────

@pytest.fixture
def random_pcd(tmp_path):
    """200 uniformly random points saved as a temp .ply file.

    Returns:
        str: Absolute path to the generated .ply file.

    Used by:
        - test_loader.py (load / validate / format-detection tests)
    """
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (200, 3)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    path = str(tmp_path / "random.ply")
    o3d.io.write_point_cloud(path, pcd)
    return path


@pytest.fixture
def flat_floor_pcd():
    """Dense horizontal floor plane (3000 pts) centred at Z≈0 with σ=0.005 noise.

    Geometry:
        X ∈ [0, 5], Y ∈ [0, 6], Z ∈ N(0, 0.005)

    Returns:
        o3d.geometry.PointCloud: In-memory cloud (no file on disk).

    Used by:
        - test_ransac.py (single-plane extraction tests)
        - test_preprocessor.py (normal estimation on a known surface)
    """
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 5, 3000)
    y = rng.uniform(0, 6, 3000)
    z = rng.normal(0, 0.005, 3000)
    pts = np.column_stack([x, y, z]).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd



@pytest.fixture
def synthetic_room_pcd(tmp_path):
    """Minimal synthetic room with known geometry for integration tests.

    Layout:
        - Floor   (2000 pts): Z ≈ 0.0, 5m × 6m
        - Ceiling (2000 pts): Z ≈ 3.0, 5m × 6m
        - Wall X=0 (1500 pts): vertical plane
        - Wall X=5 (1500 pts): vertical plane
        - Desk    (600 pts):  box at X∈[1,2.5], Y∈[1,2], Z∈[0,0.75]

    Returns:
        tuple[str, o3d.geometry.PointCloud]:
            (path_to_ply, in_memory_cloud)

    Used by:
        - test_pipeline.py (end-to-end integration tests)
        - test_exporter.py (PLY merge + JSON report tests)
    """
    rng = np.random.default_rng(99)
    pts = []

    def _plane(n, x0, x1, y0, y1, z):
        x = rng.uniform(x0, x1, n)
        y = rng.uniform(y0, y1, n)
        z_ = np.full(n, z) + rng.normal(0, 0.005, n)
        return np.column_stack([x, y, z_])

    def _vplane(n, x_fix, y0, y1, z0, z1):
        x = np.full(n, x_fix) + rng.normal(0, 0.005, n)
        y = rng.uniform(y0, y1, n)
        z = rng.uniform(z0, z1, n)
        return np.column_stack([x, y, z])

    def _box(n, x0, x1, y0, y1, z0, z1):
        x = rng.uniform(x0, x1, n)
        y = rng.uniform(y0, y1, n)
        z = rng.uniform(z0, z1, n)
        return np.column_stack([x, y, z])

    pts.append(_plane(2000, 0, 5, 0, 6, 0.0))   # floor
    pts.append(_plane(2000, 0, 5, 0, 6, 3.0))   # ceiling
    pts.append(_vplane(1500, 0, 0, 6, 0, 3))    # wall X=0
    pts.append(_vplane(1500, 5, 0, 6, 0, 3))    # wall X=5
    pts.append(_box(600,  1, 2.5, 1, 2, 0, 0.75))  # desk

    all_pts = np.vstack(pts).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    path = str(tmp_path / "synthetic_room.ply")
    o3d.io.write_point_cloud(path, pcd)
    return path, pcd


# ── Config fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def base_cfg():
    """Baseline pipeline config dict with relaxed thresholds for small test clouds.

    Key differences from production (configs/default.yaml):
        - voxel_size=0.05 (larger → fewer points → faster tests)
        - min_plane_size=100  (production=1000; test clouds are small)
        - min_cluster_points=20  (production=100)
        - orient_k=5  (production=15; avoids errors on tiny clouds)
        - output paths are empty strings (tests override via tmp_path)

    Returns:
        dict: Full pipeline config ready for any module constructor.
    """
    return {
        "preprocessing": {
            "voxel_size": 0.05,
            "sor": {"nb_neighbors": 10, "std_ratio": 2.0},
            "normal_estimation": {"radius": 0.1, "max_nn": 30, "orient_k": 5},
        },
        "ransac": {
            "distance_threshold": 0.02,
            "ransac_n": 3,
            "num_iterations": 500,
            "min_plane_size": 100,
            "max_planes": 6,
            "remaining_points_min": 50,
        },
        "dbscan": {
            "eps": 0.15,
            "min_points": 15,
            "min_cluster_points": 20,
            "max_object_size": 3.5,
        },
        "labeling": {
            "floor_z_threshold": 0.15,
            "ceiling_z_fraction": 0.8,
            "horizontal_angle_deg": 15,
            "vertical_angle_deg": 75,
        },
        "output": {
            "ply": "",
            "report": "",
            "screenshot": "",
        },
    }


# ── Cluster / plane factories ─────────────────────────────────────────────────

def make_plane(normal, centroid_z, n_pts=200):
    """Factory: creates a plane-result dict matching IterativeRANSAC output shape.

    Args:
        normal:     3-element list [nx, ny, nz] — the plane's surface normal.
        centroid_z: Median Z value of the plane (controls floor/ceiling/wall classification).
        n_pts:      Number of random points in the inlier cloud.

    Returns:
        dict with keys: plane_model, inlier_cloud, inlier_count, normal, centroid_z.

    Used by:
        - test_semantic_labeler.py (plane label assignment tests)
        - test_exporter.py (report serialisation tests)
    """
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 1, (n_pts, 3)).astype(np.float32)
    pts[:, 2] += centroid_z
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return {
        "plane_model": [*normal, 0.0],
        "inlier_cloud": pcd,
        "inlier_count": n_pts,
        "normal": list(normal),
        "centroid_z": centroid_z,
    }


def make_cluster(z_min, z_max, dims, n_pts=200):
    """Factory: creates a cluster-result dict matching DBSCANClusterer output shape.

    Args:
        z_min:  Bottom of the bounding box (controls floor-level vs. elevated labeling).
        z_max:  Top of the bounding box.
        dims:   2-element list [width, depth] — XY extent of the cluster.
        n_pts:  Number of random points in the cluster cloud.

    Returns:
        dict with keys: label_id, cloud, n_points, aabb, centroid,
                        dims (actual AABB), z_min, z_max, footprint_m2.

    Used by:
        - test_semantic_labeler.py (cluster label assignment tests)
        - test_bbox_estimator.py (OBB computation tests)
        - test_exporter.py (report serialisation tests)
    """
    rng = np.random.default_rng(8)
    x = rng.uniform(0, dims[0], n_pts)
    y = rng.uniform(0, dims[1], n_pts)
    z = rng.uniform(z_min, z_max, n_pts)
    pts = np.column_stack([x, y, z]).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    aabb = pcd.get_axis_aligned_bounding_box()
    actual_dims = (np.asarray(aabb.max_bound) - np.asarray(aabb.min_bound)).tolist()
    return {
        "label_id": 0,
        "cloud": pcd,
        "n_points": n_pts,
        "aabb": aabb,
        "centroid": pts.mean(axis=0).tolist(),
        "dims": actual_dims,
        "z_min": float(z_min),
        "z_max": float(z_max),
        "footprint_m2": float(actual_dims[0] * actual_dims[1]),
    }
