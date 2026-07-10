#!/usr/bin/env python3
"""
scripts/make_figures.py — Phase 5 figures.

Renders the two payoff figures purely from committed result JSONs — NO numbers are
hardcoded in this file. Missing inputs are skipped with a warning (so it runs at any
stage of the project).

    1. outputs/figures/label_efficiency.png  — mIoU vs number of labeled scenes
    2. outputs/figures/per_class_iou.png      — per-class IoU bars, all available arms

    python scripts/make_figures.py
"""
import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.label_spaces import S3DIS_CLASSES


def _load(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def figure_label_efficiency(path, out_png):
    data = _load(path)
    if not data or not data.get("curve"):
        return
    curve = data["curve"]
    xs = [c["n_labeled_rooms"] for c in curve]
    ys = [c["miou"] for c in curve]
    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys, "o-", color="#2b7bba", lw=2)
    plt.xlabel("Number of labeled training rooms (N)")
    plt.ylabel("Area-5 mIoU (FULL13)")
    plt.title("Label efficiency: hybrid arm mIoU vs labels")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  wrote {out_png}")


def figure_per_class(hybrid_path, geometry_path, pointnet_path, out_png):
    hybrid = _load(hybrid_path)
    geom = _load(geometry_path)
    pn = _load(pointnet_path)

    arms = {}
    if hybrid and "arms" in hybrid:
        # per_class_iou is {class: global_iou_float} (see eval_hybrid_s3dis.py)
        for arm, agg in hybrid["arms"].items():
            arms[arm] = {c: agg.get("per_class_iou", {}).get(c) for c in S3DIS_CLASSES}
    if geom and "aggregate" in geom:
        # geometry is STRUCT4 — contributes only ceiling/floor/wall
        pc = geom["aggregate"].get("per_class_iou", {})
        arms["geometry"] = {c: pc.get(c) for c in S3DIS_CLASSES}
    if pn and "per_class_iou" in pn:
        arms["pointnet++"] = {c: pn["per_class_iou"].get(c) for c in S3DIS_CLASSES}

    if not arms:
        print("  [skip] per-class figure: no arm results available")
        return

    import numpy as np
    x = np.arange(len(S3DIS_CLASSES))
    width = 0.8 / max(len(arms), 1)
    plt.figure(figsize=(13, 5))
    for i, (arm, vals) in enumerate(arms.items()):
        heights = [(vals.get(c) or 0.0) for c in S3DIS_CLASSES]
        plt.bar(x + i * width, heights, width, label=arm)
    plt.xticks(x + width * (len(arms) - 1) / 2, S3DIS_CLASSES, rotation=45, ha="right")
    plt.ylabel("IoU (Area-5)")
    plt.title("Per-class IoU by arm (geometry wins structure, learning wins objects)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"  wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Render payoff figures from result JSONs")
    ap.add_argument("--label-efficiency", default="outputs/label_efficiency.json")
    ap.add_argument("--hybrid", default="outputs/hybrid_eval.json")
    ap.add_argument("--geometry", default="outputs/eval_report.json")
    ap.add_argument("--pointnet", default="outputs/pointnet2_eval.json")
    ap.add_argument("--figdir", default="outputs/figures")
    args = ap.parse_args()

    os.makedirs(args.figdir, exist_ok=True)
    figure_label_efficiency(args.label_efficiency, os.path.join(args.figdir, "label_efficiency.png"))
    figure_per_class(args.hybrid, args.geometry, args.pointnet,
                     os.path.join(args.figdir, "per_class_iou.png"))


if __name__ == "__main__":
    main()
