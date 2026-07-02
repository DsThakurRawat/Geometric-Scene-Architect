"""
src/feature_ml.py — Feature-ML arm (Phase 2).

A thin, self-contained classifier over the geometric features already produced for
every DBSCAN cluster / RANSAC plane. Nothing new is extracted — the pipeline already
computes all of these. The point of this arm is to show that a cheap tabular model
over hand geometry can classify *objects* (the 10 non-structural S3DIS classes) that
the rule-based geometry arm lumps into a single "object" bucket.

Feature vector per cluster (order fixed by ``FEATURE_NAMES``):
    height        = dims[2]
    footprint_m2  = footprint_m2
    aspect_ratio  = dims[0] / dims[1]
    z_base        = z_min
    point_density = point_density
    n_points      = n_points
    obb_rotation  = obb_rotation_deg
    normal_z      = |normal_z|   (planes only; 0 for clusters)

Backend: RandomForestClassifier(n_estimators=200) by default; XGBoost behind a flag.
"""
from __future__ import annotations

import numpy as np
from typing import List, Optional, Union

import joblib
from sklearn.ensemble import RandomForestClassifier


FEATURE_NAMES: List[str] = [
    "height",
    "footprint_m2",
    "aspect_ratio",
    "z_base",
    "point_density",
    "n_points",
    "obb_rotation_deg",
    "normal_z",
]


def _get(obj, name, default=0.0):
    """Attribute-or-key access that works for pydantic models and plain dicts."""
    if isinstance(obj, dict):
        val = obj.get(name, default)
    else:
        val = getattr(obj, name, default)
    return default if val is None else val


def extract_cluster_features(cluster) -> np.ndarray:
    """Build the fixed-length feature vector for one ClusterResult (or dict)."""
    dims = _get(cluster, "dims", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
    w = float(dims[0]) if len(dims) > 0 else 0.0
    d = float(dims[1]) if len(dims) > 1 else 0.0
    h = float(dims[2]) if len(dims) > 2 else 0.0
    aspect_ratio = (max(w, d) / min(w, d)) if min(w, d) > 1e-6 else 1.0

    return np.array(
        [
            h,
            float(_get(cluster, "footprint_m2", 0.0)),
            aspect_ratio,
            float(_get(cluster, "z_min", 0.0)),
            float(_get(cluster, "point_density", 0.0)),
            float(_get(cluster, "n_points", 0.0)),
            float(_get(cluster, "obb_rotation_deg", 0.0) or 0.0),
            abs(float(_get(cluster, "normal_z", 0.0) or 0.0)),
        ],
        dtype=np.float64,
    )


def extract_plane_features(plane) -> np.ndarray:
    """
    Build the same fixed-length feature vector for a RANSAC PlaneResult, so the
    feature-ML-only arm can classify structural planes too. Derives geometry from
    the plane's inlier point cloud AABB; ``normal_z`` comes from the plane normal.
    """
    cloud = _get(plane, "inlier_cloud", None)
    normal = _get(plane, "normal", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
    nz = abs(float(normal[2])) if len(normal) > 2 else 0.0

    if cloud is None or len(getattr(cloud, "points", [])) == 0:
        n_pts = float(_get(plane, "inlier_count", 0.0))
        return np.array([0.0, 0.0, 1.0, float(_get(plane, "centroid_z", 0.0)),
                         0.0, n_pts, 0.0, nz], dtype=np.float64)

    pts = np.asarray(cloud.points)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    w, d, h = (mx - mn).tolist()
    aspect_ratio = (max(w, d) / min(w, d)) if min(w, d) > 1e-6 else 1.0
    footprint = float(w * d)
    volume = max(w * d * h, 1e-6)
    n_pts = float(len(pts))
    return np.array(
        [h, footprint, aspect_ratio, float(mn[2]), n_pts / volume, n_pts, 0.0, nz],
        dtype=np.float64,
    )


def build_feature_matrix(clusters: List) -> np.ndarray:
    """Stack per-cluster feature vectors into an (n, len(FEATURE_NAMES)) matrix."""
    if not clusters:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.vstack([extract_cluster_features(c) for c in clusters])


class FeatureML:
    """RandomForest / XGBoost classifier over cluster geometry features."""

    def __init__(self, backend: str = "rf", n_estimators: int = 200, random_state: int = 42):
        self.backend = backend
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names = list(FEATURE_NAMES)
        self.classes_: Optional[np.ndarray] = None
        self.model = self._make_model()

    def _make_model(self):
        if self.backend == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "xgboost backend requested but xgboost is not installed. "
                    "pip install xgboost, or use backend='rf'."
                ) from e
            return XGBClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                tree_method="hist",
                n_jobs=-1,
            )
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureML":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # XGBoost needs contiguous 0..K-1 label ids; remap and remember the mapping.
        if self.backend == "xgb":
            self._xgb_classes = np.unique(y)
            remap = {c: i for i, c in enumerate(self._xgb_classes)}
            y_fit = np.array([remap[v] for v in y])
            self.model.fit(X, y_fit)
            self.classes_ = self._xgb_classes
        else:
            self.model.fit(X, y)
            self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] == 0:
            return np.empty((0,), dtype=int)
        pred = self.model.predict(X)
        if self.backend == "xgb":
            return self._xgb_classes[np.asarray(pred).astype(int)]
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.shape[0] == 0:
            return np.empty((0, len(self.classes_) if self.classes_ is not None else 0))
        return self.model.predict_proba(X)

    def feature_importances(self) -> dict:
        """Map feature name -> importance (empty if the backend has none)."""
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return {}
        return {n: float(v) for n, v in zip(self.feature_names, imp)}

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "backend": self.backend,
                "n_estimators": self.n_estimators,
                "random_state": self.random_state,
                "feature_names": self.feature_names,
                "classes_": self.classes_,
                "model": self.model,
                "xgb_classes": getattr(self, "_xgb_classes", None),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FeatureML":
        blob = joblib.load(path)
        obj = cls(
            backend=blob["backend"],
            n_estimators=blob["n_estimators"],
            random_state=blob["random_state"],
        )
        obj.feature_names = blob["feature_names"]
        obj.classes_ = blob["classes_"]
        obj.model = blob["model"]
        if blob.get("xgb_classes") is not None:
            obj._xgb_classes = blob["xgb_classes"]
        return obj
