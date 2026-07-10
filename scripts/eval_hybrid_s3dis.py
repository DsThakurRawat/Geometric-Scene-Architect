#!/usr/bin/env python3
"""
scripts/eval_hybrid_s3dis.py — Phase 2d.

Evaluates all four arms (geometry_only, feature_ml, hybrid, hybrid_v2) on Area-5 in the
FULL13 label space, using the SAME s3dis_evaluator every arm uses (identical protocol).

Scoring uses the standard S3DIS GLOBAL protocol: one confusion matrix accumulated over ALL
Area-5 points per arm, then per-class IoU and mIoU from that matrix — NOT a mean of per-room
mIoUs (which over-weights tiny rooms and is not comparable to published S3DIS numbers). The
per-room mIoU mean/std is still reported for reference.

Fairness: geometry is computed ONCE per room and shared across all four arms, and if any arm
raises on a room that room is skipped for ALL arms — so every arm is scored on the exact same
room set (no per-arm subset drift).

Writes outputs/hybrid_eval.json and regenerates the FULL13 tables in docs/EXPERIMENTS.md.

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
from src.s3dis_evaluator import align_labels, confusion_matrix, metrics_from_confusion
from src.label_spaces import S3DIS_CLASSES

_N = len(S3DIS_CLASSES)


def arm_confusion(pred_fn, points, geo, gt):
    """Run one arm on a room's SHARED geometry -> full-res (C,C) confusion matrix."""
    pred_pts, pred_lbl = pred_fn(points, geo)
    pred_full = align_labels(pred_pts, pred_lbl, points[:, :3])
    return confusion_matrix(pred_full, gt, _N)


def summarize(cm_total, per_room_miou):
    """Global (standard S3DIS) metrics from the accumulated matrix, plus the per-room mIoU
    distribution (mean/std) for reference."""
    g = metrics_from_confusion(cm_total, S3DIS_CLASSES)
    pr = np.asarray(per_room_miou, dtype=float)
    return {
        "miou": g["miou"],                      # GLOBAL mIoU — headline
        "oa": g["overall_accuracy"],
        "per_class_iou": {c: g["per_class"][c]["iou"] for c in S3DIS_CLASSES
                          if g["per_class"][c]["iou"] is not None},
        "miou_perroom_mean": round(float(pr.mean()), 4) if pr.size else 0.0,
        "miou_perroom_std": round(float(pr.std()), 4) if pr.size else 0.0,
        "n_rooms": len(per_room_miou),
    }


def write_experiments_md(path, results, n_scored, n_skipped):
    """Regenerate the FULL13 section of EXPERIMENTS.md from real numbers only (global mIoU).
    NOTE: this OVERWRITES the file; the RGB-ablation and label-efficiency scripts re-append
    their sections afterwards, so re-run those two after this to rebuild the full document."""
    coverage = f"All four arms scored on the same {n_scored} Area-5 rooms"
    coverage += f" ({n_skipped} skipped)." if n_skipped else "."
    lines = [
        "# EXPERIMENTS — S3DIS Area-5 (FULL13)",
        "",
        "Real numbers only, produced by this repo's scripts and scored at full resolution by",
        "the shared `src/s3dis_evaluator.py`. **mIoU is the standard S3DIS global protocol**:",
        "one confusion matrix over all Area-5 points per arm, per-class IoU from it, then the",
        "mean over present classes (NOT a mean of per-room mIoUs). Do not edit by hand.",
        "",
        coverage,
        "",
        "## Arm comparison (global mIoU)",
        "",
        "| Arm | mIoU | OA |",
        "|---|---|---|",
    ]
    for arm, agg in results.items():
        lines.append(f"| {arm} | {agg['miou']:.4f} | {agg['oa']:.4f} |")
    lines += ["", "## Per-class IoU (global)", "",
              "| class | " + " | ".join(results.keys()) + " |",
              "|---|" + "|".join(["---"] * len(results)) + "|"]
    for cls in S3DIS_CLASSES:
        row = [cls]
        for agg in results.values():
            v = agg["per_class_iou"].get(cls)
            row.append(f"{v:.4f}" if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Phase 2 eval: 4 arms, FULL13, Area-5 (global mIoU)")
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
    # plus a generic 'object' bucket -> clutter; it cannot name object sub-classes (score 0).
    _clutter = S3DIS_CLASSES.index("clutter")
    _geom_to_full13 = {"ceiling": S3DIS_CLASSES.index("ceiling"), "floor": S3DIS_CLASSES.index("floor"),
                       "wall": S3DIS_CLASSES.index("wall"), "object": _clutter, "clutter": _clutter}

    def geometry_only(points, geo):
        pts, pred_str = predictor.predict_from_geo(geo)
        lbl = np.array([_geom_to_full13.get(str(s), _clutter) for s in pred_str], dtype=np.int64)
        return pts, lbl

    # Every arm takes (points, geo) and reuses the SAME room geometry.
    arms = {
        "geometry_only": geometry_only,                                        # free structure, no object names
        "feature_ml": lambda pts, geo: hybrid.predict_feature_ml_only(pts, geo=geo),
        "hybrid": lambda pts, geo: hybrid.predict_hybrid(pts, geo=geo),
        "hybrid_v2": lambda pts, geo: hybrid.predict_hybrid_v2(pts, geo=geo),  # trust geometry: floor/ceiling only
    }

    cm_total = {arm: np.zeros((_N, _N), dtype=np.int64) for arm in arms}
    per_room_miou = {arm: [] for arm in arms}
    skipped = []
    n_scored = 0

    for i, (room_name, points, gt13) in enumerate(iter_area(args.test_area, args.processed_dir)):
        if args.limit is not None and i >= args.limit:
            break
        # Compute geometry once, then all four arms. If ANY arm fails, drop the whole room
        # for ALL arms so the comparison stays on a common room set.
        try:
            geo = predictor.run_geometry(points)
            gt = gt13.astype(np.int64)
            room_cm = {arm: arm_confusion(fn, points, geo, gt) for arm, fn in arms.items()}
        except Exception as e:
            skipped.append(room_name)
            print(f"  [skip-room] {room_name}: {e}", flush=True)
            continue
        for arm, cm in room_cm.items():
            cm_total[arm] += cm
            per_room_miou[arm].append(metrics_from_confusion(cm, S3DIS_CLASSES)["miou"])
        n_scored += 1
        print(f"  [{i:3d}] {room_name:<24} hybrid mIoU(room)={per_room_miou['hybrid'][-1]:.4f}", flush=True)

    if n_scored == 0:
        print("No rooms evaluated. Did you run scripts/prepare_s3dis.py?")
        sys.exit(1)

    results = {arm: summarize(cm_total[arm], per_room_miou[arm]) for arm in arms}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"label_space": "FULL13", "test_area": args.test_area,
                   "protocol": "global_confusion_miou",
                   "n_rooms_scored": n_scored, "n_rooms_skipped": len(skipped),
                   "skipped_rooms": skipped, "arms": results}, f, indent=2)
    write_experiments_md(args.experiments_md, results, n_scored, len(skipped))

    print("\n" + "=" * 60)
    print(f"  Phase 2 — FULL13, Area-5 (global mIoU) — {n_scored} rooms, {len(skipped)} skipped")
    print("=" * 60)
    for arm, agg in results.items():
        print(f"  {arm:<12}: mIoU={agg['miou']:.4f}  OA={agg['oa']:.4f}  "
              f"(per-room {agg['miou_perroom_mean']:.4f}+/-{agg['miou_perroom_std']:.4f})")
    print(f"\n  Wrote {args.out} and {args.experiments_md}")


if __name__ == "__main__":
    main()
