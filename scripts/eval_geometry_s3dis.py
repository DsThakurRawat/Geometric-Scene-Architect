#!/usr/bin/env python3
"""
scripts/eval_geometry_s3dis.py — PHASE 1 GATE.

Runs the geometry-only arm over every Area-5 room, scores it against real
ground truth at FULL resolution in the STRUCT4 label space, and writes
outputs/eval_report.json.

This is the GO / NO-GO gate. It prints the numbers and STOPS. It writes NO
numbers to the README. A human decides whether the thesis holds before any
learned arm is built.

    python scripts/eval_geometry_s3dis.py                 # all Area-5 rooms
    python scripts/eval_geometry_s3dis.py --limit 5       # quick smoke test
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3dis_loader import iter_area
from src.point_predictor import PointPredictor
from src.s3dis_evaluator import align_labels, compute_metrics, print_report
from src.label_spaces import STRUCT4, to_struct4, map_string_to_struct4_idx


def pred_strings_to_struct4(pred_labels: np.ndarray) -> np.ndarray:
    """Vectorised map of geometry string labels -> STRUCT4 integer ids."""
    return np.array([map_string_to_struct4_idx(str(s)) for s in pred_labels], dtype=np.int64)


def evaluate_room(predictor: PointPredictor, points: np.ndarray, gt13: np.ndarray) -> dict:
    """Geometry -> align -> remap to STRUCT4 -> point-level metrics for one room."""
    pred_pts, pred_str = predictor.predict(points)
    # Lift the downsampled prediction back onto every full-res GT point.
    pred_full_str = align_labels(pred_pts, pred_str, points[:, :3])
    pred_full = pred_strings_to_struct4(pred_full_str)
    gt4 = to_struct4(gt13)
    return compute_metrics(pred_full, gt4, STRUCT4)


def aggregate(per_room: list) -> dict:
    """Mean +/- std of mIoU / OA and per-class IoU across rooms (present classes only)."""
    mious = np.array([r["metrics"]["miou"] for r in per_room], dtype=float)
    oas = np.array([r["metrics"]["overall_accuracy"] for r in per_room], dtype=float)

    per_class = {}
    for cls in STRUCT4:
        vals = [
            r["metrics"]["per_class"][cls]["iou"]
            for r in per_room
            if r["metrics"]["per_class"].get(cls, {}).get("iou") is not None
        ]
        if vals:
            per_class[cls] = {
                "iou_mean": round(float(np.mean(vals)), 4),
                "iou_std": round(float(np.std(vals)), 4),
                "n_rooms": len(vals),
            }
    return {
        "miou_mean": round(float(mious.mean()), 4),
        "miou_std": round(float(mious.std()), 4),
        "oa_mean": round(float(oas.mean()), 4),
        "oa_std": round(float(oas.std()), 4),
        "per_class_iou": per_class,
        "n_rooms": len(per_room),
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 1 gate: geometry-only STRUCT4 eval on Area-5")
    ap.add_argument("--test-area", type=int, default=5)
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="outputs/eval_report.json")
    ap.add_argument("--limit", type=int, default=None, help="only first N rooms (smoke test)")
    args = ap.parse_args()

    predictor = PointPredictor()
    per_room = []

    for i, (room_name, points, gt13) in enumerate(iter_area(args.test_area, args.processed_dir)):
        if args.limit is not None and i >= args.limit:
            break
        try:
            m = evaluate_room(predictor, points, gt13)
        except Exception as e:  # never let one bad room abort the gate
            print(f"  [skip] {room_name}: {e}")
            continue
        per_room.append({"room": room_name, "metrics": m})
        print(f"  [{i:3d}] {room_name:<24} mIoU={m['miou']:.4f} OA={m['overall_accuracy']:.4f}")

    if not per_room:
        print("No rooms evaluated. Did you run scripts/prepare_s3dis.py?")
        sys.exit(1)

    agg = aggregate(per_room)
    report = {
        "arm": "geometry_only",
        "label_space": "STRUCT4",
        "test_area": args.test_area,
        "resolution": "full",
        "aggregate": agg,
        "per_room": per_room,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 66)
    print("  PHASE 1 GATE — geometry-only, STRUCT4, Area-5, full resolution")
    print("=" * 66)
    print(f"  Rooms evaluated : {agg['n_rooms']}")
    print(f"  mIoU            : {agg['miou_mean']:.4f} +/- {agg['miou_std']:.4f}")
    print(f"  Overall Acc     : {agg['oa_mean']:.4f} +/- {agg['oa_std']:.4f}")
    print("  Per-class IoU (mean +/- std):")
    for cls, s in agg["per_class_iou"].items():
        print(f"    {cls:<10}: {s['iou_mean']:.4f} +/- {s['iou_std']:.4f}  (n={s['n_rooms']})")
    print("=" * 66)
    print(f"\n  Report written to {args.out}")
    print("  GATE: review floor/wall/ceiling IoU vs the ~88/97/70 reference band.")
    print("  GO if structure is strong on real scans; NO-GO -> fix geometry first.")
    print("  >>> STOPPING for human review. Do NOT auto-proceed to the arms. <<<\n")


if __name__ == "__main__":
    main()
