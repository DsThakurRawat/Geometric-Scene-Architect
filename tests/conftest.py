"""
conftest.py — shared pytest fixtures for all test modules.
"""
import pytest
import numpy as np
import open3d as o3d


# ── Basic clouds ─────────────────────────────────────────────────────────────

@pytest.fixture
def random_pcd(tmp_path):
    """200 random points, written to a temp .ply."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1, (200, 3)).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    path = str(tmp_path / "random.ply")
    o3d.io.write_point_cloud(path, pcd)
    return path


@pytest.fixture
def flat_floor_pcd():
    """Dense horizontal floor plane at Z~0 — used by test_ransac.py."""
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
    """
    Minimal synthetic room: floor + ceiling + 2 walls + 1 furniture box.
    Written to a temp .ply and returned as (path, pcd) tuple.
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
