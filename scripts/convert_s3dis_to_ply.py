#!/usr/bin/env python3
"""
scripts/convert_s3dis_to_ply.py

Converts a raw S3DIS room directory (with Annotations/ folder) into
a single colored .ply file usable by the segmentation pipeline.

Usage:
    python scripts/convert_s3dis_to_ply.py \
        --room_path data/raw/Area_1/office_1 \
        --output    data/processed/office_1.ply
"""
import os
import argparse
import numpy as np
import open3d as o3d


def merge_room_to_ply(room_path: str, out_path: str):
    """
    Merge all annotation .txt files in a room's Annotations/ directory
    into a single colored .ply file.

    S3DIS format per .txt file: X Y Z R G B (space-separated, one point per line).
    """
    ann_dir = os.path.join(room_path, "Annotations")
    if not os.path.isdir(ann_dir):
        # Try flat room directory (some preprocessed S3DIS versions)
        ann_dir = room_path

    all_pts = []
    for fname in sorted(os.listdir(ann_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(ann_dir, fname)
        try:
            pts = np.loadtxt(fpath)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[1] < 6:
                # XYZ only, no color
                pad = np.full((pts.shape[0], 3), 128.0)
                pts = np.hstack([pts[:, :3], pad])
            all_pts.append(pts[:, :6])
            print(f"  Loaded {len(pts):>8,} pts from {fname}")
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")

    if not all_pts:
        raise RuntimeError(f"No .txt files found in {ann_dir}")

    cloud = np.vstack(all_pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(cloud[:, 3:6] / 255.0)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"\nSaved {out_path}: {len(pcd.points):,} total points.")


def main():
    parser = argparse.ArgumentParser(description="Convert S3DIS room to .ply")
    parser.add_argument("--room_path", type=str, required=True,
                        help="Path to S3DIS room folder (containing Annotations/ subfolder).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write the output .ply file.")
    args = parser.parse_args()
    merge_room_to_ply(args.room_path, args.output)


if __name__ == "__main__":
    main()
