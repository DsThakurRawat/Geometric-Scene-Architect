"""Integration tests for the geometry wrappers used by every S3DIS arm.

Uses the synthetic_room_pcd fixture (floor/ceiling/walls/desk with known geometry)
so these run without the S3DIS download.
"""
import numpy as np
import pytest

from src.point_predictor import PointPredictor
from src.hybrid_labeler import HybridLabeler
from src.feature_ml import FeatureML, FEATURE_NAMES
from src.s3dis_evaluator import align_labels, compute_metrics
from src.label_spaces import STRUCT4, S3DIS_CLASSES, map_string_to_struct4_idx


@pytest.fixture(scope="module")
def predictor():
    return PointPredictor()


def _room_points(pcd):
    return np.asarray(pcd.points, dtype=np.float64)


def test_predict_returns_aligned_shapes(predictor, synthetic_room_pcd):
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    used, labels = predictor.predict(pts)
    assert used.shape[0] == labels.shape[0]
    assert used.shape[1] == 3
    assert used.shape[0] > 0


def test_predict_finds_structure(predictor, synthetic_room_pcd):
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    used, labels = predictor.predict(pts)
    label_set = set(np.unique(labels).tolist())
    # a clean synthetic room should yield floor and at least one wall/ceiling
    assert "floor" in label_set
    assert label_set & {"wall", "ceiling"}


def test_run_geometry_exposes_segments(predictor, synthetic_room_pcd):
    _, pcd = synthetic_room_pcd
    geo = predictor.run_geometry(_room_points(pcd))
    segs = predictor.segments(geo)
    assert len(segs) > 0
    for s in segs:
        assert s["features"].shape == (len(FEATURE_NAMES),)
        assert s["indices"].max() < len(geo["clean_pts"])
        assert s["mean_rgb"].shape == (3,)


def test_predict_then_align_scores_reasonably(predictor, synthetic_room_pcd):
    """End-to-end: predict -> align to full res -> STRUCT4 metrics. Floor IoU should be high."""
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    used, labels = predictor.predict(pts)
    pred_full = align_labels(used, labels, pts[:, :3])
    pred_idx = np.array([map_string_to_struct4_idx(str(s)) for s in pred_full])

    # Build a GT for the synthetic room from known z-structure:
    # floor z~0, ceiling z~3, else object. (Walls span z so approximate by x-extremes.)
    z = pts[:, 2]
    gt = np.full(len(pts), STRUCT4.index("object"), dtype=int)
    gt[z < 0.2] = STRUCT4.index("floor")
    gt[z > 2.8] = STRUCT4.index("ceiling")
    m = compute_metrics(pred_idx, gt, STRUCT4)
    # floor is the easiest — expect strong IoU
    assert m["per_class"]["floor"]["iou"] is not None
    assert m["per_class"]["floor"]["iou"] > 0.5


def test_hybrid_labeler_emits_full13(predictor, synthetic_room_pcd):
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    # tiny dummy model over the FEATURE_NAMES space
    rng = np.random.RandomState(0)
    X = rng.rand(20, len(FEATURE_NAMES))
    y = rng.randint(0, len(S3DIS_CLASSES), size=20)
    model = FeatureML(backend="rf", n_estimators=10).fit(X, y)

    hybrid = HybridLabeler(model, predictor)
    used, pred = hybrid.predict_hybrid(pts)
    assert used.shape[0] == pred.shape[0]
    assert pred.dtype.kind in "iu"
    assert pred.min() >= 0 and pred.max() < len(S3DIS_CLASSES)

    used2, pred2 = hybrid.predict_feature_ml_only(pts)
    assert used2.shape[0] == pred2.shape[0]


def test_hybrid_v2_hands_vertical_planes_to_model(predictor, synthetic_room_pcd):
    """hybrid_v2 trusts geometry only for floor/ceiling; every vertical plane goes to the
    model. With a model that always predicts 'door', v2 yields zero geometry-forced walls,
    whereas plain hybrid keeps geometry's wall calls. Guards the door/window-cannibalisation
    fix that motivated hybrid_v2."""
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    door = S3DIS_CLASSES.index("door")
    wall = S3DIS_CLASSES.index("wall")
    X = np.random.RandomState(0).rand(20, len(FEATURE_NAMES))
    model = FeatureML(backend="rf", n_estimators=10).fit(X, np.full(20, door))

    hybrid = HybridLabeler(model, predictor)
    _, v2 = hybrid.predict_hybrid_v2(pts)
    _, hy = hybrid.predict_hybrid(pts)

    # v2 hands every vertical plane to the model (which only knows 'door'); no wall survives.
    assert (v2 == wall).sum() == 0
    assert (v2 == door).sum() > 0
    # plain hybrid trusts geometry's structural calls, so it keeps >= as many walls as v2.
    assert (hy == wall).sum() >= (v2 == wall).sum()


def test_run_geometry_stable_structure(predictor, synthetic_room_pcd):
    """Two run_geometry calls yield the same downsampled cloud size and the same number of
    segments. NOTE: exact per-point segment membership is byte-identical only under
    OMP_NUM_THREADS=1 — Open3D's RANSAC is OpenMP-nondeterministic (ties broken per-thread),
    so scripts pin one thread via src/__init__ for reproducible numbers. The pytest process
    loads Open3D (via conftest) before src, so it may run multithreaded; this asserts the
    coarse structure, which is stable regardless."""
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    g1, g2 = predictor.run_geometry(pts), predictor.run_geometry(pts)
    assert len(g1["clean_pts"]) == len(g2["clean_pts"]) > 0
    assert len(predictor.segments(g1)) == len(predictor.segments(g2)) > 0


def test_geometry_reuse_paths_agree(predictor, synthetic_room_pcd):
    """Given ONE geometry result, every reuse entry point agrees — the real invariant the
    4-arm eval and label-efficiency sweep rely on (they run geometry once per room and share
    it across arms, so cross-arm identity holds by construction, not by RANSAC determinism)."""
    _, pcd = synthetic_room_pcd
    pts = _room_points(pcd)
    geo = predictor.run_geometry(pts)

    # geometry_only reuse returns exactly this geo's cloud
    used_g, lbl_g = predictor.predict_from_geo(geo)
    assert np.array_equal(used_g, geo["clean_pts"])
    assert used_g.shape[0] == lbl_g.shape[0]

    X = np.random.RandomState(0).rand(30, len(FEATURE_NAMES))
    y = np.random.RandomState(1).randint(0, len(S3DIS_CLASSES), size=30)
    model = FeatureML(backend="rf", n_estimators=10).fit(X, y)
    hybrid = HybridLabeler(model, predictor)

    # predict_hybrid(geo=) and predict_hybrid_from_segments must agree on the same geometry
    _, reuse = hybrid.predict_hybrid(pts, geo=geo)
    _, seg = hybrid.predict_hybrid_from_segments(geo["clean_pts"], predictor.segments(geo))
    assert np.array_equal(reuse, seg)
