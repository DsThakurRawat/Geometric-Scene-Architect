#!/usr/bin/env python3
"""
scripts/eval_pointnet2_s3dis.py — Phase 3 scoring.

Scores PointNet++ Area-5 predictions with the SHARED evaluator (standard S3DIS global
mIoU, FULL13), so the PN++ row is apples-to-apples with geometry_only / feature_ml /
hybrid / hybrid_v2 in outputs/hybrid_eval.json.

PN++ is trained on Colab (see colab/pointnet2_s3dis.ipynb) and exports per-point Area-5
predictions at FULL resolution, in each room's original points.npy order, as an .npz:
    {room_name: (N,) int FULL13 labels}
This script accumulates one global confusion matrix over all Area-5 points and writes
outputs/pointnet2_eval.json (schema consumed by scripts/make_figures.py).

    python scripts/eval_pointnet2_s3dis.py --preds outputs/pointnet2_area5_preds.npz
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3dis_evaluator import confusion_matrix, metrics_from_confusion
from src.label_spaces import S3DIS_CLASSES

_N = len(S3DIS_CLASSES)


def main():
    ap = argparse.ArgumentParser(description="Phase 3: score PointNet++ Area-5 preds (global mIoU)")
    ap.add_argument("--preds", default="outputs/pointnet2_area5_preds.npz",
                    help="npz of {room_name: (N,) FULL13 int preds}, full-res, original order")
    ap.add_argument("--test-area", type=int, default=5)
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="outputs/pointnet2_eval.json")
    args = ap.parse_args()

    if not os.path.exists(args.preds):
        print(f"Predictions not found at {args.preds}. Run the Colab notebook and download "
              f"the exported preds first (see colab/pointnet2_s3dis.ipynb).")
        sys.exit(1)

    preds = np.load(args.preds)
    pred_rooms = set(preds.files)
    area_dir = os.path.join(args.processed_dir, f"Area_{args.test_area}")
    room_names = sorted(d for d in os.listdir(area_dir)
                        if os.path.isdir(os.path.join(area_dir, d)))

    cm_total = np.zeros((_N, _N), dtype=np.int64)
    scored, skipped, mismatched = [], [], []

    for name in room_names:
        if name not in pred_rooms:
            skipped.append(name)
            print(f"  [skip] {name}: no prediction in {args.preds}", flush=True)
            continue
        gt = np.load(os.path.join(area_dir, name, "labels.npy")).astype(np.int64)
        pr = np.asarray(preds[name]).astype(np.int64).reshape(-1)
        if pr.shape[0] != gt.shape[0]:
            mismatched.append(name)
            print(f"  [skip] {name}: preds {pr.shape[0]} != gt {gt.shape[0]} points", flush=True)
            continue
        cm_total += confusion_matrix(pr, gt, _N)
        scored.append(name)
        print(f"  scored {name} ({gt.shape[0]} pts)", flush=True)

    if not scored:
        print("No rooms scored — check the preds file and room-name keys.")
        sys.exit(1)

    g = metrics_from_confusion(cm_total, S3DIS_CLASSES)
    report = {
        "arm": "pointnet++",
        "label_space": "FULL13",
        "test_area": args.test_area,
        "protocol": "global_confusion_miou",
        "n_rooms_scored": len(scored),
        "n_rooms_skipped": len(skipped) + len(mismatched),
        "skipped_rooms": skipped,
        "mismatched_rooms": mismatched,
        "miou": g["miou"],
        "oa": g["overall_accuracy"],
        # top-level {class: float} so make_figures picks it up as the pointnet++ arm
        "per_class_iou": {c: g["per_class"][c]["iou"] for c in S3DIS_CLASSES
                          if g["per_class"][c]["iou"] is not None},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  Phase 3 — PointNet++, FULL13, Area-5 (global mIoU) — "
          f"{len(scored)} rooms, {len(skipped) + len(mismatched)} skipped")
    print("=" * 60)
    print(f"  mIoU={g['miou']:.4f}  OA={g['overall_accuracy']:.4f}")
    for c in S3DIS_CLASSES:
        iou = report["per_class_iou"].get(c)
        if iou is not None:
            print(f"    {c:<10}: {iou:.4f}")
    print(f"\n  Wrote {args.out}")
    print("  Re-run scripts/make_figures.py --figdir docs/images to add PN++ to the per-class figure.")


if __name__ == "__main__":
    main()
