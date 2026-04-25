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


def generate_cylinder_pts(n, center_xy, radius, z_range, noise=0.005, seed=0):
    """Generates random points on the surface of a vertical cylinder (e.g., table leg, column)."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(*z_range, n)
    x = center_xy[0] + radius * np.cos(theta) + rng.normal(0, noise, n)
    y = center_xy[1] + radius * np.sin(theta) + rng.normal(0, noise, n)
    return np.column_stack([x, y, z])


# ── Room layouts ─────────────────────────────────────────────────────────────

def create_rectangular_room(W, L, H, density, seed_start=0):
    """Standard rectangular room: floor, ceiling, 4 walls."""
    pts = []
    seed = seed_start

    # Structural surfaces
    pts.append(generate_plane_pts(density, (0, W), (0, L), 0.0, seed=seed));      seed += 1  # floor
    pts.append(generate_plane_pts(density, (0, W), (0, L), H, seed=seed));        seed += 1  # ceiling
    pts.append(generate_vertical_plane_pts(density, (0, 0), (0, L), (0, H), seed=seed));    seed += 1  # wall X=0
    pts.append(generate_vertical_plane_pts(density, (W, W), (0, L), (0, H), seed=seed));    seed += 1  # wall X=W
    pts.append(generate_vertical_plane_pts(density, (0, W), (0, 0), (0, H), seed=seed));    seed += 1  # wall Y=0
    pts.append(generate_vertical_plane_pts(density, (0, W), (L, L), (0, H), seed=seed));    seed += 1  # wall Y=L

    return pts, seed


def create_l_shaped_room(W, L, H, density, seed_start=0):
    """L-shaped room: main rectangle + extension, creating an indoor corner."""
    pts = []
    seed = seed_start

    # Main room (full width, half length)
    half_L = L * 0.6
    pts.append(generate_plane_pts(density, (0, W), (0, half_L), 0.0, seed=seed));   seed += 1  # floor main
    pts.append(generate_plane_pts(density, (0, W), (0, half_L), H, seed=seed));     seed += 1  # ceiling main

    # Extension (half width, remaining length)
    half_W = W * 0.5
    pts.append(generate_plane_pts(density // 2, (0, half_W), (half_L, L), 0.0, seed=seed));  seed += 1  # floor ext
    pts.append(generate_plane_pts(density // 2, (0, half_W), (half_L, L), H, seed=seed));    seed += 1  # ceiling ext

    # Outer walls
    pts.append(generate_vertical_plane_pts(density, (0, 0), (0, L), (0, H), seed=seed));           seed += 1  # X=0 full
    pts.append(generate_vertical_plane_pts(density, (W, W), (0, half_L), (0, H), seed=seed));      seed += 1  # X=W main
    pts.append(generate_vertical_plane_pts(density // 2, (half_W, half_W), (half_L, L), (0, H), seed=seed)); seed += 1  # X=half_W ext
    pts.append(generate_vertical_plane_pts(density, (0, W), (0, 0), (0, H), seed=seed));           seed += 1  # Y=0
    pts.append(generate_vertical_plane_pts(density // 2, (0, half_W), (L, L), (0, H), seed=seed)); seed += 1  # Y=L ext

    # Inner corner wall (the "step" of the L)
    pts.append(generate_vertical_plane_pts(density // 2, (half_W, W), (half_L, half_L), (0, H), seed=seed)); seed += 1

    return pts, seed


# ── Furniture presets ────────────────────────────────────────────────────────

def add_standard_furniture(pts, W, L, seed_start=0):
    """Adds a variety of furniture items to stress-test the labeler."""
    seed = seed_start

    # Desk (table-height, medium footprint)
    pts.append(generate_box_pts(5000, (1.0, 2.5), (1.0, 2.0), (0.00, 0.75), seed=seed));  seed += 1

    # Chair (shorter, smaller footprint)
    pts.append(generate_box_pts(2000, (1.0, 1.5), (2.3, 2.7), (0.00, 0.45), seed=seed));  seed += 1

    # Monitor on desk (elevated small object)
    pts.append(generate_box_pts(2000, (1.5, 2.0), (1.2, 1.5), (0.75, 1.10), seed=seed));  seed += 1

    # Wardrobe (tall furniture)
    pts.append(generate_box_pts(8000, (3.5, 4.5), (4.0, 5.5), (0.00, 2.20), seed=seed));  seed += 1

    # Bookshelf (tall, narrow depth)
    pts.append(generate_box_pts(6000, (0.1, 0.4), (2.0, 3.5), (0.00, 1.80), seed=seed));  seed += 1

    # Small side table
    pts.append(generate_box_pts(1500, (3.0, 3.5), (1.0, 1.5), (0.00, 0.50), seed=seed));  seed += 1

    # Ceiling light (high fixture)
    pts.append(generate_box_pts(1000, (2.2, 2.8), (2.7, 3.3), (2.80, 3.00), seed=seed));  seed += 1

    # Wall-mounted shelf (floating, narrow)
    pts.append(generate_box_pts(1500, (4.0, 4.8), (0.1, 0.3), (1.20, 1.40), seed=seed));  seed += 1

    # Floor lamp (tall, tiny footprint - cylinder-like)
    pts.append(generate_cylinder_pts(1500, (4.7, 1.5), 0.1, (0.0, 1.6), seed=seed));  seed += 1

    return pts, seed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic 3D room point cloud.")
    parser.add_argument("--output",  type=str,   default="data/synthetic/room_01.ply",
                        help="Path to write the output .ply file.")
    parser.add_argument("--W",       type=float, default=5.0,   help="Room width  (X, metres).")
    parser.add_argument("--L",       type=float, default=6.0,   help="Room length (Y, metres).")
    parser.add_argument("--H",       type=float, default=3.0,   help="Room height (Z, metres).")
    parser.add_argument("--density", type=int,   default=10000, help="Points per wall/floor/ceiling surface.")
    parser.add_argument("--layout",  type=str,   default="rectangular",
                        choices=["rectangular", "l-shaped"],
                        help="Room layout type.")
    parser.add_argument("--no-noise", action="store_true",
                        help="Disable floating noise points (for ground-truth evaluation).")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Generate room structure ───────────────────────────────────────────
    if args.layout == "l-shaped":
        pts, seed = create_l_shaped_room(args.W, args.L, args.H, args.density)
    else:
        pts, seed = create_rectangular_room(args.W, args.L, args.H, args.density)

    # ── Add furniture ─────────────────────────────────────────────────────
    pts, seed = add_standard_furniture(pts, args.W, args.L, seed_start=seed)

    # ── Floating noise (scan artifacts) ───────────────────────────────────
    if not args.no_noise:
        rng = np.random.default_rng(seed)
        pts.append(rng.uniform(0, max(args.W, args.L, args.H), (500, 3)))

    all_pts = np.vstack(pts).astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # uniform grey (no RGB info in raw scan)

    o3d.io.write_point_cloud(args.output, pcd)
    print(f"Generated synthetic room → {args.output}  ({len(all_pts):,} points)")
    print(f"  Layout: {args.layout}")
    print(f"  Dimensions: {args.W}m × {args.L}m × {args.H}m")
    print(f"  Furniture items: 9")
    if not args.no_noise:
        print(f"  Noise points: 500")


if __name__ == "__main__":
    main()
