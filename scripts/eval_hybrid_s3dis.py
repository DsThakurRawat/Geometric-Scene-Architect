#!/usr/bin/env python3
"""
scripts/eval_hybrid_s3dis.py — Phase 2d.

Evaluates the feature-ML-only and hybrid arms on Area-5 in the FULL13 label space,
using the SAME s3dis_evaluator every other arm uses (identical protocol). Writes
outputs/hybrid_eval.json and refreshes the real-numbers tables in docs/EXPERIMENTS.md.

    python scripts/eval_hybrid_s3dis.py
    python scripts/eval_hybrid_s3dis.py --limit 5      # smoke test
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.s3dis_loader import iter_area
from src.point_predictor import PointPredictor
from src.hybrid_labeler import HybridLabeler
from src.feature_ml import FeatureML
from src.s3dis_evaluator import align_labels, compute_metrics
from src.label_spaces import S3DIS_CLASSES


def eval_arm(predict_fn, points, gt13):
    pred_pts, pred_lbl = predict_fn(points)
    pred_full = align_labels(pred_pts, pred_lbl, points[:, :3])
    return compute_metrics(pred_full, gt13.astype(np.int64), S3DIS_CLASSES)


def aggregate(per_room):
    mious = np.array([r["miou"] for r in per_room], dtype=float)
    oas = np.array([r["overall_accuracy"] for r in per_room], dtype=float)
    per_class = {}
    for cls in S3DIS_CLASSES:
        vals = [r["per_class"][cls]["iou"] for r in per_room
                if r["per_class"].get(cls, {}).get("iou") is not None]
        if vals:
            per_class[cls] = {"iou_mean": round(float(np.mean(vals)), 4), "n_rooms": len(vals)}
    return {
        "miou_mean": round(float(mious.mean()), 4),
        "miou_std": round(float(mious.std()), 4),
        "oa_mean": round(float(oas.mean()), 4),
        "per_class_iou": per_class,
        "n_rooms": len(per_room),
    }


def write_experiments_md(path, results):
    """Regenerate the FULL13 results section of EXPERIMENTS.md from real numbers only."""
    lines = [
        "# EXPERIMENTS — S3DIS Area-5 (FULL13)",
        "",
        "All numbers below are produced by `scripts/eval_hybrid_s3dis.py` in this repo,",
        "scored at full resolution by the shared `src/s3dis_evaluator.py`. Do not edit by hand.",
        "",
        "## Arm comparison (mean over Area-5 rooms)",
        "",
        "| Arm | mIoU | OA |",
        "|---|---|---|",
    ]
    for arm, agg in results.items():
        lines.append(f"| {arm} | {agg['miou_mean']:.4f} | {agg['oa_mean']:.4f} |")
    lines += ["", "## Per-class IoU (mean over rooms)", "",
              "| class | " + " | ".join(results.keys()) + " |",
              "|---|" + "|".join(["---"] * len(results)) + "|"]
    for cls in S3DIS_CLASSES:
        row = [cls]
        for agg in results.values():
            v = agg["per_class_iou"].get(cls, {}).get("iou_mean")
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Phase 2 eval: feature-ML + hybrid, FULL13, Area-5")
    ap.add_argument("--model", default="models/feature_ml.pkl")
    ap.add_argument("--test-area", type=int, default=5)
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="outputs/hybrid_eval.json")
    ap.add_argument("--experiments-md", default="docs/EXPERIMENTS.md")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Run scripts/train_feature_ml.py first.")
        sys.exit(1)

    model = FeatureML.load(args.model)
    predictor = PointPredictor()
    hybrid = HybridLabeler(model, predictor)

    # geometry-only arm scored in the SAME FULL13 space: geometry emits ceiling/floor/wall
    # plus a generic 'object' bucket -> clutter; it cannot name object sub-classes (they score 0).
    _clutter = S3DIS_CLASSES.index("clutter")
    _geom_to_full13 = {"ceiling": S3DIS_CLASSES.index("ceiling"), "floor": S3DIS_CLASSES.index("floor"),
                       "wall": S3DIS_CLASSES.index("wall"), "object": _clutter, "clutter": _clutter}

    def geometry_only(points):
        pts, pred_str = predictor.predict(points)
        lbl = np.array([_geom_to_full13.get(str(s), _clutter) for s in pred_str], dtype=np.int64)
        return pts, lbl

    arms = {
        "geometry_only": geometry_only,          # free structure, cannot name objects
        "feature_ml": hybrid.predict_feature_ml_only,
        "hybrid": hybrid.predict_hybrid,
        "hybrid_v2": hybrid.predict_hybrid_v2,   # trust geometry for floor/ceiling only
    }
    per_room = {arm: [] for arm in arms}

    for i, (room_name, points, gt13) in enumerate(iter_area(args.test_area, args.processed_dir)):
        if args.limit is not None and i >= args.limit:
            break
        for arm, fn in arms.items():
            try:
                m = eval_arm(fn, points, gt13)
                per_room[arm].append(m)
            except Exception as e:
                print(f"  [skip] {arm} {room_name}: {e}")
        if per_room["hybrid"]:
            print(f"  [{i:3d}] {room_name:<24} hybrid mIoU={per_room['hybrid'][-1]['miou']:.4f}")

    results = {arm: aggregate(rooms) for arm, rooms in per_room.items() if rooms}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"label_space": "FULL13", "test_area": args.test_area,
                   "arms": results}, f, indent=2)
    write_experiments_md(args.experiments_md, results)

    print("\n" + "=" * 60)
    print("  Phase 2 — FULL13, Area-5")
    print("=" * 60)
    for arm, agg in results.items():
        print(f"  {arm:<12}: mIoU={agg['miou_mean']:.4f}+/-{agg['miou_std']:.4f}  OA={agg['oa_mean']:.4f}")
    print(f"\n  Wrote {args.out} and {args.experiments_md}")


if __name__ == "__main__":
    main()
