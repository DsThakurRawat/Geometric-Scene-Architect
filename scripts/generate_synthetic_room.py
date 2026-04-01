import numpy as np
import open3d as o3d
import argparse
import os


# ── Point generators ────────────────────────────────────────────────────────

def generate_plane_pts(n, x_range, y_range, z_val, noise=0.01, seed=0):
    """Generates random points for a horizontal plane."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)
    y = rng.uniform(*y_range, n)
    z = np.full(n, z_val) + rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


def generate_vertical_plane_pts(n, x_range, y_range, z_range, noise=0.01, seed=0):
    """Generates random points for a vertical plane (wall)."""
    rng = np.random.default_rng(seed)
    if x_range[0] == x_range[1]:   # Plane in YZ
        x = np.full(n, x_range[0]) + rng.normal(0, noise, n)
        y = rng.uniform(*y_range, n)
        z = rng.uniform(*z_range, n)
    else:                           # Plane in XZ
        x = rng.uniform(*x_range, n)
        y = np.full(n, y_range[0]) + rng.normal(0, noise, n)
        z = rng.uniform(*z_range, n)
    return np.column_stack([x, y, z])


def generate_box_pts(n, x_range, y_range, z_range, noise=0.005, seed=0):
    """Generates random points inside a volumetric box (furniture)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(*x_range, n)
    y = rng.uniform(*y_range, n)
    z = rng.uniform(*z_range, n)
    return np.column_stack([x, y, z]) + rng.normal(0, noise, (n, 3))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic 3D room point cloud.")
    parser.add_argument("--output",  type=str,   default="data/synthetic/room_01.ply",
                        help="Path to write the output .ply file.")
    parser.add_argument("--W",       type=float, default=5.0,   help="Room width  (X, metres).")
    parser.add_argument("--L",       type=float, default=6.0,   help="Room length (Y, metres).")
    parser.add_argument("--H",       type=float, default=3.0,   help="Room height (Z, metres).")
    parser.add_argument("--density", type=int,   default=10000, help="Points per wall/floor/ceiling surface.")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    pts = []
    seed = 0  # Incremented per surface so each has a unique random sequence

    # ── Structural surfaces ───────────────────────────────────────────────────
    pts.append(generate_plane_pts(args.density, (0, args.W), (0, args.L), 0.0, seed=seed))
    seed += 1  # floor
    pts.append(generate_plane_pts(args.density, (0, args.W), (0, args.L), args.H, seed=seed))
    seed += 1  # ceiling
    pts.append(generate_vertical_plane_pts(args.density, (0, 0), (0, args.L), (0, args.H), seed=seed))
    seed += 1  # wall X=0
    pts.append(generate_vertical_plane_pts(args.density, (args.W, args.W), (0, args.L), (0, args.H), seed=seed))
    seed += 1  # wall X=W
    pts.append(generate_vertical_plane_pts(args.density, (0, args.W), (0, 0), (0, args.H), seed=seed))
    seed += 1  # wall Y=0
    pts.append(generate_vertical_plane_pts(args.density, (0, args.W), (args.L, args.L), (0, args.H), seed=seed))
    seed += 1  # wall Y=L

    # ── Furniture ─────────────────────────────────────────────────────────────
    pts.append(generate_box_pts(5000, (1.0, 2.5), (1.0, 2.0), (0.00, 0.75), seed=seed))
    seed += 1  # desk
    pts.append(generate_box_pts(2000, (1.5, 2.0), (1.2, 1.5), (0.75, 1.10), seed=seed))
    seed += 1  # monitor on desk
    pts.append(generate_box_pts(8000, (3.5, 4.5), (4.0, 5.5), (0.00, 2.20), seed=seed))
    seed += 1  # wardrobe
    pts.append(generate_box_pts(1000, (2.2, 2.8), (2.7, 3.3), (2.80, 3.00), seed=seed))
    seed += 1  # ceiling light

    # ── Floating noise ────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    pts.append(rng.uniform(0, max(args.W, args.L, args.H), (500, 3)))

    all_pts = np.vstack(pts).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # uniform grey (no RGB info in raw scan)

    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Generated synthetic room → {args.output}  ({len(all_pts):,} points)")


if __name__ == "__main__":
    main()
