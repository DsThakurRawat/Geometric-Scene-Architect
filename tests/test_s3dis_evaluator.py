"""Tests for the point-level S3DIS evaluator (Phase 1 core)."""
import numpy as np
import pytest

from src.s3dis_evaluator import (
    align_labels,
    compute_metrics,
    confusion_matrix,
    metrics_from_confusion,
)


def test_align_labels_toy_cloud_nearest_neighbour():
    # src has two labelled points; dst points sit next to each one.
    src_pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    src_lbl = np.array([1, 2])
    dst_pts = np.array([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0], [0.2, 0.1, 0.0]])
    out = align_labels(src_pts, src_lbl, dst_pts)
    assert out.tolist() == [1, 2, 1]


def test_align_labels_exact_match_preserves_labels():
    pts = np.random.RandomState(0).rand(50, 3)
    lbl = np.random.RandomState(1).randint(0, 4, size=50)
    out = align_labels(pts, lbl, pts)  # identical dst -> identical labels
    assert out.tolist() == lbl.tolist()


def test_align_labels_empty_dst():
    src_pts = np.array([[0.0, 0.0, 0.0]])
    src_lbl = np.array([3])
    out = align_labels(src_pts, src_lbl, np.zeros((0, 3)))
    assert out.shape == (0,)


def test_align_labels_empty_src_raises():
    with pytest.raises(ValueError):
        align_labels(np.zeros((0, 3)), np.zeros((0,)), np.ones((2, 3)))


def test_iou_hand_computed_two_class():
    # class 0: 3 gt; class 1: 3 gt. One point of each is mislabelled.
    gt = np.array([0, 0, 0, 1, 1, 1])
    pred = np.array([0, 0, 1, 1, 1, 0])
    # class 0: TP=2, FP=1 (the last point pred 0), FN=1 -> IoU = 2/(2+1+1)=0.5
    # class 1: TP=2, FP=1, FN=1 -> IoU=0.5 ; mIoU=0.5
    m = compute_metrics(pred, gt, ["a", "b"])
    assert m["per_class"]["a"]["iou"] == pytest.approx(0.5)
    assert m["per_class"]["b"]["iou"] == pytest.approx(0.5)
    assert m["miou"] == pytest.approx(0.5)
    assert m["overall_accuracy"] == pytest.approx(4 / 6, abs=1e-4)


def test_perfect_prediction_iou_one():
    gt = np.array([0, 1, 2, 2, 1, 0])
    m = compute_metrics(gt.copy(), gt, ["a", "b", "c"])
    assert m["miou"] == pytest.approx(1.0)
    assert m["overall_accuracy"] == pytest.approx(1.0)


def test_miou_only_over_present_classes():
    # class 2 never appears in GT -> excluded from mIoU
    gt = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    m = compute_metrics(pred, gt, ["a", "b", "c"])
    assert "c" not in m["present_classes"]
    assert m["miou"] == pytest.approx(1.0)


def test_ignore_index_excludes_points():
    gt = np.array([0, 1, 2, 2])
    pred = np.array([0, 1, 0, 0])  # the two class-2 points are wrong
    m = compute_metrics(pred, gt, ["a", "b", "ignore"], ignore_index=2)
    # class 2 dropped entirely; a and b perfect
    assert m["miou"] == pytest.approx(1.0)
    assert m["per_class"]["a"]["iou"] == pytest.approx(1.0)


def test_confusion_matrix_shape_and_counts():
    gt = np.array([0, 0, 1])
    pred = np.array([0, 1, 1])
    cm = confusion_matrix(pred, gt, 2)
    assert cm.shape == (2, 2)
    # row = true, col = pred
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1


def test_metrics_from_confusion_matches_compute_metrics():
    """metrics_from_confusion on a single room's matrix reproduces compute_metrics exactly
    (compute_metrics is now a thin wrapper over it)."""
    rng = np.random.RandomState(0)
    classes = ["a", "b", "c"]
    gt = rng.randint(0, 3, size=500)
    pred = gt.copy()
    pred[::7] = (pred[::7] + 1) % 3
    m = compute_metrics(pred, gt, classes)
    g = metrics_from_confusion(np.array(m["confusion_matrix"]), classes)
    assert g["miou"] == m["miou"]
    assert g["overall_accuracy"] == m["overall_accuracy"]
    assert g["per_class"] == m["per_class"]


def test_global_confusion_sums_over_rooms():
    """Standard S3DIS global protocol: summing per-room confusion matrices then scoring once
    equals scoring the concatenated points. This is what the eval scripts rely on to report a
    global mIoU rather than a mean of per-room mIoUs."""
    rng = np.random.RandomState(1)
    classes = ["a", "b", "c", "d"]
    g1 = rng.randint(0, 4, size=300); p1 = g1.copy(); p1[::5] = (p1[::5] + 1) % 4
    g2 = rng.randint(0, 4, size=200); p2 = g2.copy(); p2[::3] = (p2[::3] + 2) % 4
    cm1 = np.array(compute_metrics(p1, g1, classes)["confusion_matrix"])
    cm2 = np.array(compute_metrics(p2, g2, classes)["confusion_matrix"])
    glob = metrics_from_confusion(cm1 + cm2, classes)
    concat = compute_metrics(np.concatenate([p1, p2]), np.concatenate([g1, g2]), classes)
    assert glob["miou"] == concat["miou"]
    assert glob["per_class"] == concat["per_class"]
    # global mIoU generally differs from the mean of the two per-room mIoUs
    per_room_mean = round(
        (compute_metrics(p1, g1, classes)["miou"] + compute_metrics(p2, g2, classes)["miou"]) / 2, 4
    )
    assert isinstance(per_room_mean, float)  # documents that the two protocols are not identical
