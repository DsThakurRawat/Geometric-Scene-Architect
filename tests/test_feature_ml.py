"""Tests for the feature-ML arm: feature extraction, majority vote, model roundtrip."""
import os
import numpy as np
import pytest

from src.feature_ml import (
    FeatureML,
    FEATURE_NAMES,
    extract_cluster_features,
    build_feature_matrix,
)


def _cluster_dict(**kw):
    base = dict(
        dims=[1.0, 0.5, 0.8],
        footprint_m2=0.5,
        z_min=0.0,
        point_density=1000.0,
        n_points=300,
        obb_rotation_deg=12.0,
        normal_z=0.0,
    )
    base.update(kw)
    return base


def test_feature_vector_length_and_order():
    f = extract_cluster_features(_cluster_dict())
    assert f.shape == (len(FEATURE_NAMES),)
    # height = dims[2], footprint, aspect = max(1,0.5)/min(1,0.5)=2.0, z_base=0
    assert f[0] == pytest.approx(0.8)   # height
    assert f[1] == pytest.approx(0.5)   # footprint
    assert f[2] == pytest.approx(2.0)   # aspect ratio
    assert f[3] == pytest.approx(0.0)   # z_base


def test_aspect_ratio_guard_on_zero_dim():
    f = extract_cluster_features(_cluster_dict(dims=[0.0, 0.0, 0.5]))
    assert f[2] == pytest.approx(1.0)   # no division by zero


def test_build_feature_matrix_shape():
    clusters = [_cluster_dict(), _cluster_dict(dims=[2.0, 2.0, 2.0])]
    X = build_feature_matrix(clusters)
    assert X.shape == (2, len(FEATURE_NAMES))


def test_build_feature_matrix_empty():
    X = build_feature_matrix([])
    assert X.shape == (0, len(FEATURE_NAMES))


def test_fit_predict_and_importances():
    rng = np.random.RandomState(0)
    # two separable blobs -> RF should learn them perfectly
    X0 = rng.normal(0, 0.1, size=(40, len(FEATURE_NAMES)))
    X1 = rng.normal(5, 0.1, size=(40, len(FEATURE_NAMES)))
    X = np.vstack([X0, X1])
    y = np.array([7] * 40 + [8] * 40)  # table vs chair ids
    model = FeatureML(backend="rf", n_estimators=50)
    model.fit(X, y)
    assert set(model.predict(X).tolist()) <= {7, 8}
    assert model.predict(X).tolist() == y.tolist()
    imp = model.feature_importances()
    assert len(imp) == len(FEATURE_NAMES)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.RandomState(1)
    X = rng.normal(0, 1, size=(30, len(FEATURE_NAMES)))
    y = rng.randint(0, 3, size=30)
    model = FeatureML(backend="rf", n_estimators=20)
    model.fit(X, y)
    path = os.path.join(tmp_path, "m.pkl")
    model.save(path)
    loaded = FeatureML.load(path)
    assert np.array_equal(loaded.predict(X), model.predict(X))
    assert loaded.feature_names == FEATURE_NAMES


def test_majority_vote_logic():
    # mirrors the majority-vote used in train_feature_ml
    seg_gt = np.array([8, 8, 8, 2, 2])  # chair chair chair wall wall
    n_classes = 13
    maj = int(np.bincount(seg_gt, minlength=n_classes).argmax())
    assert maj == 8
