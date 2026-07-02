#!/usr/bin/env python3
"""
scripts/train_feature_ml.py — Phase 2b.

Builds the feature-ML training set from real GT + geometry, fits the model, reports
train/val accuracy and feature importances, and saves models/feature_ml.pkl.

For every room in the train areas (1-4,6):
    1. run the shared geometry (planes + clusters),
    2. downsample the full-res GT onto the clean cloud by nearest neighbour,
    3. give each geometry segment its majority-vote GT label,
    4. accumulate (8-d feature vector, S3DIS class) pairs.

    python scripts/train_feature_ml.py                     # RandomForest
    python scripts/train_feature_ml.py --backend xgb       # XGBoost
    python scripts/train_feature_ml.py --limit-per-area 5  # quick smoke test
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.s3dis_loader import iter_area
from src.point_predictor import PointPredictor
from src.s3dis_evaluator import align_labels
from src.feature_ml import FeatureML, FEATURE_NAMES
from src.label_spaces import S3DIS_CLASSES


def majority_label(labels: np.ndarray) -> int:
    return int(np.bincount(labels, minlength=len(S3DIS_CLASSES)).argmax())


def build_training_set(predictor, train_areas, processed_dir, limit_per_area=None):
    X_all, y_all = [], []
    for area in train_areas:
        n = 0
        for room_name, points, gt13 in iter_area(area, processed_dir):
            if limit_per_area is not None and n >= limit_per_area:
                break
            n += 1
            try:
                geo = predictor.run_geometry(points)
                clean_pts = geo["clean_pts"]
                # downsample GT onto the clean cloud (NN from full-res GT points)
                gt_on_clean = align_labels(points[:, :3], gt13, clean_pts)
                for seg in predictor.segments(geo):
                    seg_gt = gt_on_clean[seg["indices"]]
                    if seg_gt.size == 0:
                        continue
                    X_all.append(seg["features"])
                    y_all.append(majority_label(seg_gt))
            except Exception as e:
                print(f"  [skip] Area_{area}/{room_name}: {e}")
                continue
            print(f"  Area_{area}/{room_name}: {len(X_all)} segments so far")
    if not X_all:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,), dtype=int)
    return np.vstack(X_all), np.array(y_all, dtype=int)


def main():
    ap = argparse.ArgumentParser(description="Train the feature-ML arm on S3DIS train areas")
    ap.add_argument("--backend", choices=["rf", "xgb"], default="rf")
    ap.add_argument("--train-areas", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--processed-dir", default="data/s3dis/processed")
    ap.add_argument("--out", default="models/feature_ml.pkl")
    ap.add_argument("--limit-per-area", type=int, default=None)
    args = ap.parse_args()

    predictor = PointPredictor()
    print(f"Building training set from areas {args.train_areas} ...")
    X, y = build_training_set(predictor, args.train_areas, args.processed_dir, args.limit_per_area)
    if X.shape[0] == 0:
        print("No training segments produced. Did you run prepare_s3dis.py?")
        sys.exit(1)

    print(f"\nTotal segments: {X.shape[0]}  features: {X.shape[1]}")
    classes, counts = np.unique(y, return_counts=True)
    print("Class distribution (majority-vote GT per segment):")
    for c, n in zip(classes, counts):
        print(f"  {S3DIS_CLASSES[c]:<12}: {n}")

    strat = y if np.min(np.bincount(y)) >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat
    )

    model = FeatureML(backend=args.backend)
    model.fit(X_tr, y_tr)

    tr_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    print(f"\nTrain accuracy: {tr_acc:.4f}   Val accuracy: {val_acc:.4f}")

    imp = model.feature_importances()
    if imp:
        print("\nFeature importances:")
        for name, val in sorted(imp.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {name:<18}: {val:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"\nModel saved to {args.out}")


if __name__ == "__main__":
    main()
