#!/usr/bin/env python3
"""
scripts/pack_for_colab.py — package the processed S3DIS cache for Colab (Phase 3).

The processed cache is ~7.2 GB (points float32 (N,6) + labels int32, 544 files). This packs
it into one compressed .npz per area with a compact encoding — xyz float32, rgb uint8, label
uint8 (~43% smaller, 6 files) — that the Colab PointNet++ notebook reads directly.

Per-area npz keys are "<room>::xyz", "<room>::rgb", "<room>::label".

    python scripts/pack_for_colab.py                 # all areas -> data/s3dis/colab_pack/
    python scripts/pack_for_colab.py --limit-per-area 2   # quick round-trip check

Then upload data/s3dis/colab_pack/*.npz to Google Drive and run colab/pointnet2_s3dis.ipynb.
"""
import os
import sys
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Pack processed S3DIS into per-area npz for Colab")
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out-dir", default="data/s3dis/colab_pack")
    ap.add_argument("--areas", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--limit-per-area", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total_bytes = 0
    for area in args.areas:
        area_dir = os.path.join(args.processed_dir, f"Area_{area}")
        if not os.path.isdir(area_dir):
            print(f"  [skip] {area_dir} missing")
            continue
        rooms = sorted(d for d in os.listdir(area_dir)
                       if os.path.isdir(os.path.join(area_dir, d)))
        if args.limit_per_area is not None:
            rooms = rooms[:args.limit_per_area]

        packed = {}
        for room in rooms:
            pts = np.load(os.path.join(area_dir, room, "points.npy"))          # (N,6) f32
            lbl = np.load(os.path.join(area_dir, room, "labels.npy"))          # (N,) int
            packed[f"{room}::xyz"] = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
            packed[f"{room}::rgb"] = np.clip(pts[:, 3:6], 0, 255).astype(np.uint8)
            packed[f"{room}::label"] = lbl.astype(np.uint8)

        out = os.path.join(args.out_dir, f"Area_{area}.npz")
        np.savez_compressed(out, **packed)
        sz = os.path.getsize(out)
        total_bytes += sz
        print(f"  Area_{area}: {len(rooms)} rooms -> {out} ({sz/1e6:.1f} MB)", flush=True)

    print(f"\nPacked {total_bytes/1e9:.2f} GB total into {args.out_dir}/")
    print("Upload these .npz files to Google Drive, then run colab/pointnet2_s3dis.ipynb.")


if __name__ == "__main__":
    main()
