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
import time
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


def build_room_segments(predictor, area, processed_dir, limit=None, for_training=False):
    """Precompute geometry per room once, keeping only what its role needs.

    Holding the full ``geo`` (cKDTree + Open3D plane/cluster clouds) and raw points for
    every pool AND test room at once exhausts RAM (the sweep OOM-killed at ~95 rooms).
    So each room keeps only its role's essentials:

    - ``for_training=True`` (the labeled pool): the trainer only reads ``segs`` +
      ``gt_on_clean`` — drop geo/tree/point-clouds/raw points entirely.
    - ``for_training=False`` (the fixed Area-5 test set): eval predicts from the precomputed
      ``segs`` + ``clean_pts`` (deterministic, so identical to recomputing geometry) and needs
      full-res xyz + ``gt13`` to upsample and score. The heavy geo (cKDTree, Open3D
      plane/cluster clouds) is dropped once segments are extracted; xyz is stored float32 and
      gt13 int32 to stay well under the box's small free-RAM headroom.
    """
    rooms = []
    n = 0
    for room_name, points, gt13 in iter_area(area, processed_dir):
        if limit is not None and n >= limit:
            break
        n += 1
        t0 = time.time()
        try:
            geo = predictor.run_geometry(points)
            gt_on_clean = align_labels(points[:, :3], gt13, geo["clean_pts"])
            if for_training:
                rooms.append({"name": room_name,
                              "segs": predictor.segments(geo),
                              "gt_on_clean": gt_on_clean})
            else:
                rooms.append({"name": room_name,
                              "points": np.ascontiguousarray(points[:, :3], dtype=np.float32),
                              "gt13": gt13.astype(np.int32),
                              "clean_pts": geo["clean_pts"].astype(np.float32),
                              "segs": predictor.segments(geo)})
            # Per-room progress (flushed): the biggest S3DIS rooms take ~40s in
            # orient_normals, so a silent precompute of 68 rooms looks hung.
            print(f"  Area_{area}/{room_name} [{n}]: {len(geo['clean_pts'])} pts "
                  f"{time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"  [skip] Area_{area}/{room_name}: {e}", flush=True)
    return rooms


def train_on_rooms(predictor, rooms):
    X, y = [], []
    for r in rooms:
        for seg in r["segs"]:
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
    cm_total = np.zeros((_N, _N), dtype=np.int64)
    for r in test_rooms:
        # Predict from the segments precomputed in build_room_segments (deterministic, so
        # identical to recomputing) instead of re-running ~15s/room of RANSAC/DBSCAN for
        # every N in the sweep — and without holding the heavy geo in RAM.
        pred_pts, pred = hybrid.predict_hybrid_from_segments(r["clean_pts"], r["segs"])
        pred_full = align_labels(pred_pts, pred, r["points"])
        cm_total += confusion_matrix(pred_full, r["gt13"].astype(np.int64), _N)
    # Standard S3DIS global mIoU (one confusion matrix over all Area-5 points).
    return metrics_from_confusion(cm_total, S3DIS_CLASSES)["miou"] if cm_total.sum() else 0.0


def main():
    ap = argparse.ArgumentParser(description="Phase 5 label-efficiency sweep")
    ap.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40])
    ap.add_argument("--train-area", type=int, default=1,
                    help="pool of labeled rooms to subsample from")
    ap.add_argument("--test-area", type=int, default=5)
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="outputs/label_efficiency.json")
    ap.add_argument("--experiments-md", default="docs/EXPERIMENTS.md")
    ap.add_argument("--limit", type=int, default=None, help="cap test rooms (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    predictor = PointPredictor()
    print("Precomputing training pool geometry ...", flush=True)
    pool = build_room_segments(predictor, args.train_area, args.processed_dir, for_training=True)
    print("Precomputing test geometry ...", flush=True)
    test_rooms = build_room_segments(predictor, args.test_area, args.processed_dir,
                                     args.limit, for_training=False)

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
        print(f"  N={n:3d} labeled rooms -> Area-5 hybrid mIoU={miou:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"arm": "hybrid", "label_space": "FULL13", "test_area": args.test_area,
                   "train_pool_area": args.train_area, "curve": curve}, f, indent=2)

    # append (not overwrite) a label-efficiency section to EXPERIMENTS.md
    section = ["",
               f"## Label efficiency (hybrid arm, FULL13, Area-5; global mIoU, train pool = Area-{args.train_area})",
               "",
               f"Hybrid global mIoU on Area-5 vs number of labeled training rooms N (single random "
               f"subset per N, seed {args.seed}). See `docs/images/label_efficiency.png`.",
               "", "| N labeled rooms | mIoU |", "|---|---|"]
    for c in curve:
        section.append(f"| {c['n_labeled_rooms']} | {c['miou']:.4f} |")
    section.append("")
    os.makedirs(os.path.dirname(args.experiments_md), exist_ok=True)
    with open(args.experiments_md, "a") as f:
        f.write("\n".join(section))
    print(f"\nWrote {args.out} and appended label-efficiency section to {args.experiments_md}")


if __name__ == "__main__":
    main()
