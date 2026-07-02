"""
src/s3dis_evaluator.py — Point-level segmentation evaluator for S3DIS (Phase 1 core).

This is the piece every arm depends on. It does two things the segment-level
``src/evaluator.py`` cannot:

1. **align_labels** — propagate predictions made on a *downsampled* cloud to the
   *full-resolution* ground-truth points via nearest-neighbour lookup. Scoring is
   always done at full resolution (the standard S3DIS protocol). Never score the
   downsampled prediction against downsampled GT — that silently inflates metrics.

2. **compute_metrics** — per-class IoU = TP / (TP + FP + FN); mIoU = mean over the
   classes actually *present in the GT*; plus per-class precision/recall/F1, overall
   accuracy, and a confusion matrix. Everything is computed on integer label arrays
   with numpy so it scales to the millions of points in an S3DIS room.

The two failure modes this module exists to prevent (both silently corrupt metrics):
    - Downsample alignment: always align pred -> full-res GT before scoring.
    - Taxonomy mapping: remap BOTH pred and GT into the active label space
      (STRUCT4 for the geometry arm, FULL13 for the learned arms) before scoring.
      That remap happens in the calling script; this module scores whatever
      integer arrays it is given against the ``class_names`` provided.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is a hard dependency, this is a safety net
    from sklearn.neighbors import KDTree as _SkKDTree
    _HAVE_SCIPY = False


def align_labels(
    src_pts: np.ndarray, src_lbl: np.ndarray, dst_pts: np.ndarray
) -> np.ndarray:
    """
    Propagate labels from a source cloud to a destination cloud by nearest neighbour.

    For every point in ``dst_pts`` find the closest point in ``src_pts`` and copy its
    label. Used to lift predictions from the 0.03 m downsampled cloud back onto the
    full-resolution GT points so metrics are computed at full resolution.

    Args:
        src_pts: (M, 3) source coordinates (e.g. the downsampled prediction cloud).
        src_lbl: (M,)   integer label per source point.
        dst_pts: (N, 3) destination coordinates (e.g. full-res GT points).

    Returns:
        (N,) integer labels, one per destination point.
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    src_lbl = np.asarray(src_lbl)

    if src_pts.shape[0] == 0:
        raise ValueError("align_labels: src_pts is empty, cannot propagate labels.")
    if src_pts.shape[0] != src_lbl.shape[0]:
        raise ValueError(
            f"align_labels: src_pts ({src_pts.shape[0]}) and src_lbl "
            f"({src_lbl.shape[0]}) length mismatch."
        )
    if dst_pts.shape[0] == 0:
        return np.empty((0,), dtype=src_lbl.dtype)

    if _HAVE_SCIPY:
        tree = cKDTree(src_pts)
        _, idx = tree.query(dst_pts, k=1)
    else:  # pragma: no cover
        tree = _SkKDTree(src_pts)
        idx = tree.query(dst_pts, k=1, return_distance=False)[:, 0]

    idx = np.asarray(idx).reshape(-1)
    return src_lbl[idx]


def confusion_matrix(
    pred: np.ndarray, gt: np.ndarray, num_classes: int
) -> np.ndarray:
    """
    Fast integer confusion matrix ``cm[true, pred]`` via a single bincount.

    Labels outside ``[0, num_classes)`` are ignored (e.g. an ignore_index that has
    already been masked out upstream). Mirrors the logic of
    ``src/evaluator.py::confusion_matrix`` but operates on integer arrays for speed.
    """
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    if pred.shape != gt.shape:
        raise ValueError(f"confusion_matrix: pred {pred.shape} vs gt {gt.shape} mismatch.")

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    gt_v = gt[valid].astype(np.int64)
    pred_v = pred[valid].astype(np.int64)

    flat = gt_v * num_classes + pred_v
    cm = np.bincount(flat, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    class_names: List[str],
    ignore_index: Optional[int] = None,
) -> Dict:
    """
    Point-level segmentation metrics.

    Args:
        pred: (N,) integer predicted labels in ``[0, len(class_names))``.
        gt:   (N,) integer ground-truth labels in the same space.
        class_names: ordered list defining the label space (index == class id).
        ignore_index: optional class id to exclude from *all* metrics (points whose
            GT equals this are dropped before scoring).

    Returns:
        dict with keys:
            per_class: {name: {iou, precision, recall, f1, support, tp, fp, fn}}
            miou: mean IoU over classes present in GT (ignore_index excluded)
            overall_accuracy: fraction of points classified correctly
            present_classes: names of classes with support > 0 in GT
            confusion_matrix: (C, C) list-of-lists, rows=true, cols=pred
            n_points: number of scored points
    """
    n_classes = len(class_names)
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)

    if ignore_index is not None:
        keep = gt != ignore_index
        pred = pred[keep]
        gt = gt[keep]

    cm = confusion_matrix(pred, gt, n_classes)

    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)      # true occurrences per class
    predicted = cm.sum(axis=0).astype(np.float64)    # predicted occurrences per class
    fp = predicted - tp
    fn = support - tp

    per_class: Dict[str, Dict] = {}
    ious = []
    present = []
    for i, name in enumerate(class_names):
        if ignore_index is not None and i == ignore_index:
            continue
        denom_iou = tp[i] + fp[i] + fn[i]
        iou = float(tp[i] / denom_iou) if denom_iou > 0 else float("nan")
        precision = float(tp[i] / (tp[i] + fp[i])) if (tp[i] + fp[i]) > 0 else 0.0
        recall = float(tp[i] / (tp[i] + fn[i])) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[name] = {
            "iou": round(iou, 4) if not np.isnan(iou) else None,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support[i]),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
        }
        # mIoU averages over classes PRESENT in the ground truth only (standard S3DIS).
        if support[i] > 0:
            present.append(name)
            if not np.isnan(iou):
                ious.append(iou)

    total = float(cm.sum())
    overall_accuracy = float(tp.sum() / total) if total > 0 else 0.0
    miou = float(np.mean(ious)) if ious else 0.0

    return {
        "per_class": per_class,
        "miou": round(miou, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "present_classes": present,
        "confusion_matrix": cm.astype(int).tolist(),
        "n_points": int(total),
    }


def print_report(metrics: Dict, class_names: List[str], title: str = "S3DIS Evaluation") -> None:
    """Human-readable per-class IoU table."""
    print(f"\n{'=' * 66}")
    print(f"  {title}")
    print(f"{'=' * 66}")
    print(f"  Points scored   : {metrics['n_points']:,}")
    print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"  mIoU (present)  : {metrics['miou']:.4f}")
    print(f"{'-' * 66}")
    print(f"  {'class':<14}{'IoU':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'support':>14}")
    print(f"{'-' * 66}")
    for name in class_names:
        m = metrics["per_class"].get(name)
        if m is None:
            continue
        iou = m["iou"]
        iou_s = f"{iou:.4f}" if iou is not None else "  n/a "
        print(
            f"  {name:<14}{iou_s:>8}{m['precision']:>8.4f}{m['recall']:>8.4f}"
            f"{m['f1']:>8.4f}{m['support']:>14,}"
        )
    print(f"{'=' * 66}\n")
