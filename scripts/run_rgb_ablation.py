#!/usr/bin/env python3
"""
scripts/run_rgb_ablation.py — Phase 4 modality sweep.

Trains and evaluates the feature-ML arm twice — geometry features only (xyz) vs.
geometry features + mean segment colour (xyz+rgb) — and reports the FULL13 Area-5
mIoU/OA delta. The geometry-only rule arm is the no-RGB floor and is not retrained
here (see eval_geometry_s3dis.py). Appends a real-numbers table to EXPERIMENTS.md.

    python scripts/run_rgb_ablation.py
    python scripts/run_rgb_ablation.py --limit-per-area 5 --limit 5   # smoke test
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
from src.s3dis_evaluator import align_labels, compute_metrics
from src.feature_ml import FeatureML
from src.label_spaces import S3DIS_CLASSES

_CLUTTER = S3DIS_CLASSES.index("clutter")


def seg_features(seg, use_rgb):
    return np.concatenate([seg["features"], seg["mean_rgb"]]) if use_rgb else seg["features"]


def collect(predictor, areas, processed_dir, use_rgb, limit_per_area=None, keep_full=True):
    """Return list of (per-room segment dicts) so we can reuse geometry across variants.

    Training only consumes ``segs`` + ``gt_on_clean`` (see train_variant); the full
    ``points``/``gt13``/``clean_pts`` arrays are only needed to upsample and score the
    TEST rooms (see eval_variant). Holding the full point clouds for every training
    room across all train areas exhausts RAM, so callers pass keep_full=False for the
    train pool to drop those heavy arrays once segments are computed.
    """
    rooms = []
    for area in areas:
        n = 0
        t_area = time.time()
        for room_name, points, gt13 in iter_area(area, processed_dir):
            if limit_per_area is not None and n >= limit_per_area:
                break
            n += 1
            # Per-room progress (flushed): geometry cost scales with room size and the
            # largest S3DIS rooms (e.g. Area_6/hallway_1, ~3.2M pts) take ~40s each in
            # orient_normals, so a per-area-only print makes a slow-but-live run look
            # hung. Print each room's wall time so big rooms are visibly progressing.
            t0 = time.time()
            try:
                geo = predictor.run_geometry(points)
                gt_on_clean = align_labels(points[:, :3], gt13, geo["clean_pts"])
                segs = predictor.segments(geo)
                rooms.append({
                    "points": points if keep_full else None,
                    "gt13": gt13 if keep_full else None,
                    "clean_pts": geo["clean_pts"] if keep_full else None,
                    "segs": segs, "gt_on_clean": gt_on_clean})
                print(f"  Area_{area}/{room_name} [{n}]: {len(geo['clean_pts'])} pts "
                      f"{time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                print(f"  [skip] Area_{area}/{room_name}: {e}", flush=True)
        print(f"  Area_{area}: {n} rooms done in {time.time() - t_area:.0f}s", flush=True)
    return rooms


def train_variant(rooms, use_rgb):
    X, y = [], []
    for r in rooms:
        for seg in r["segs"]:
            g = r["gt_on_clean"][seg["indices"]]
            if g.size == 0:
                continue
            X.append(seg_features(seg, use_rgb))
            y.append(int(np.bincount(g, minlength=len(S3DIS_CLASSES)).argmax()))
    model = FeatureML(backend="rf")
    model.fit(np.vstack(X), np.array(y))
    return model


def eval_variant(model, rooms, use_rgb):
    mious, oas = [], []
    for r in rooms:
        pred = np.full(len(r["clean_pts"]), _CLUTTER, dtype=np.int64)
        segs = r["segs"]
        if segs:
            X = np.vstack([seg_features(s, use_rgb) for s in segs])
            yhat = model.predict(X)
            for s, c in zip(segs, yhat):
                pred[s["indices"]] = int(c)
        pred_full = align_labels(r["clean_pts"], pred, r["points"][:, :3])
        m = compute_metrics(pred_full, r["gt13"].astype(np.int64), S3DIS_CLASSES)
        mious.append(m["miou"])
        oas.append(m["overall_accuracy"])
    return {"miou_mean": round(float(np.mean(mious)), 4),
            "oa_mean": round(float(np.mean(oas)), 4), "n_rooms": len(rooms)}


def main():
    ap = argparse.ArgumentParser(description="Phase 4 RGB ablation (feature-ML arm)")
    ap.add_argument("--train-areas", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--test-area", type=int, default=5)
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="outputs/rgb_ablation.json")
    ap.add_argument("--experiments-md", default="docs/EXPERIMENTS.md")
    ap.add_argument("--limit-per-area", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    predictor = PointPredictor()
    print("Collecting train geometry ...", flush=True)
    train_rooms = collect(predictor, args.train_areas, args.processed_dir, True,
                          args.limit_per_area, keep_full=False)
    print("Collecting test geometry ...", flush=True)
    test_rooms = collect(predictor, [args.test_area], args.processed_dir, True,
                         args.limit if args.limit else None, keep_full=True)

    results = {}
    for use_rgb, name in [(False, "xyz"), (True, "xyz+rgb")]:
        model = train_variant(train_rooms, use_rgb)
        results[name] = eval_variant(model, test_rooms, use_rgb)
        print(f"  {name:<8}: mIoU={results[name]['miou_mean']:.4f} OA={results[name]['oa_mean']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"arm": "feature_ml", "label_space": "FULL13",
                   "test_area": args.test_area, "variants": results}, f, indent=2)

    # append (not overwrite) an ablation section to EXPERIMENTS.md
    section = ["", "## RGB ablation (feature-ML arm, FULL13, Area-5)", "",
               "| variant | mIoU | OA |", "|---|---|---|"]
    for name, r in results.items():
        section.append(f"| {name} | {r['miou_mean']:.4f} | {r['oa_mean']:.4f} |")
    section.append("")
    os.makedirs(os.path.dirname(args.experiments_md), exist_ok=True)
    with open(args.experiments_md, "a") as f:
        f.write("\n".join(section))
    print(f"\nWrote {args.out} and appended ablation to {args.experiments_md}")


if __name__ == "__main__":
    main()
