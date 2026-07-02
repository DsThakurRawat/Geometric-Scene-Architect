#!/usr/bin/env python3
"""
scripts/run_label_efficiency.py — Phase 5 payoff experiment.

Retrains the hybrid arm with N labeled training rooms for N in a sweep, holding
Area-5 fixed as the test set, and logs FULL13 mIoU vs N. The story: geometric
structure lets the learned arm reach good accuracy with far fewer human labels.

Writes outputs/label_efficiency.json (consumed by make_figures.py).

    python scripts/run_label_efficiency.py
    python scripts/run_label_efficiency.py --n-values 1 2 5 --limit 5   # smoke test
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


def build_room_segments(predictor, area, processed_dir, limit=None):
    """Precompute geometry + downsampled GT for each room once (reused across N)."""
    rooms = []
    n = 0
    for room_name, points, gt13 in iter_area(area, processed_dir):
        if limit is not None and n >= limit:
            break
        n += 1
        try:
            geo = predictor.run_geometry(points)
            gt_on_clean = align_labels(points[:, :3], gt13, geo["clean_pts"])
            rooms.append({"name": room_name, "points": points, "gt13": gt13,
                          "geo": geo, "gt_on_clean": gt_on_clean})
        except Exception as e:
            print(f"  [skip] Area_{area}/{room_name}: {e}")
    return rooms


def train_on_rooms(predictor, rooms):
    X, y = [], []
    for r in rooms:
        for seg in predictor.segments(r["geo"]):
            g = r["gt_on_clean"][seg["indices"]]
            if g.size == 0:
                continue
            X.append(seg["features"])
            y.append(int(np.bincount(g, minlength=len(S3DIS_CLASSES)).argmax()))
    if not X:
        return None
    model = FeatureML(backend="rf")
    model.fit(np.vstack(X), np.array(y))
    return model


def eval_hybrid(predictor, model, test_rooms):
    hybrid = HybridLabeler(model, predictor)
    mious = []
    for r in test_rooms:
        # reuse precomputed geometry via a lightweight re-predict on stored points
        pred_pts, pred = hybrid.predict_hybrid(r["points"])
        pred_full = align_labels(pred_pts, pred, r["points"][:, :3])
        m = compute_metrics(pred_full, r["gt13"].astype(np.int64), S3DIS_CLASSES)
        mious.append(m["miou"])
    return float(np.mean(mious)) if mious else 0.0


def main():
    ap = argparse.ArgumentParser(description="Phase 5 label-efficiency sweep")
    ap.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40])
    ap.add_argument("--train-area", type=int, default=1,
                    help="pool of labeled rooms to subsample from")
    ap.add_argument("--test-area", type=int, default=5)
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="outputs/label_efficiency.json")
    ap.add_argument("--limit", type=int, default=None, help="cap test rooms (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    predictor = PointPredictor()
    print("Precomputing training pool geometry ...")
    pool = build_room_segments(predictor, args.train_area, args.processed_dir)
    print("Precomputing test geometry ...")
    test_rooms = build_room_segments(predictor, args.test_area, args.processed_dir, args.limit)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pool))

    curve = []
    for n in args.n_values:
        if n > len(pool):
            print(f"  N={n} exceeds pool size {len(pool)}, capping.")
            n = len(pool)
        subset = [pool[i] for i in order[:n]]
        model = train_on_rooms(predictor, subset)
        if model is None:
            continue
        miou = eval_hybrid(predictor, model, test_rooms)
        curve.append({"n_labeled_rooms": int(n), "miou": round(miou, 4)})
        print(f"  N={n:3d} labeled rooms -> Area-5 hybrid mIoU={miou:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"arm": "hybrid", "label_space": "FULL13", "test_area": args.test_area,
                   "train_pool_area": args.train_area, "curve": curve}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
