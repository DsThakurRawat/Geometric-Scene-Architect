# src/hybrid_labeler.py
"""
Hybrid arm (Phase 2c): geometry does the structure for free, learning earns its keep
on objects.

    - Structural planes labelled ceiling/floor/wall by the deterministic geometry rules
      are trusted directly (these are the classes geometry already nails).
    - Every remaining segment (DBSCAN clusters + non-structural planes) is classified
      into the full 13-class space by the trained FeatureML model.

Both this and the ``feature_ml_only`` variant emit per-point FULL13 integer labels on
the downsampled cloud, which ``s3dis_evaluator.align_labels`` lifts to full resolution.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from src.point_predictor import PointPredictor
from src.feature_ml import FeatureML
from src.label_spaces import S3DIS_CLASSES

_CLUTTER = S3DIS_CLASSES.index("clutter")
_STRUCT_IDX = {"ceiling": 0, "floor": 1, "wall": 2}


class HybridLabeler:
    def __init__(self, model: FeatureML, predictor: PointPredictor | None = None):
        self.model = model
        self.predictor = predictor or PointPredictor()

    def _predict(self, points: np.ndarray, trust: frozenset) -> Tuple[np.ndarray, np.ndarray]:
        """Predict FULL13 per-point labels.

        ``trust`` is the set of geometry structural labels taken directly from the geometry
        rules; every other segment is classified by the model. An empty set == feature-ML-only.
        """
        geo = self.predictor.run_geometry(points)
        clean_pts = geo["clean_pts"]
        pred = np.full(len(clean_pts), _CLUTTER, dtype=np.int64)

        segs = self.predictor.segments(geo)

        ml_segments = []
        for seg in segs:
            if seg["structural_label"] in trust:
                pred[seg["indices"]] = _STRUCT_IDX[seg["structural_label"]]
            else:
                ml_segments.append(seg)

        if ml_segments:
            X = np.vstack([s["features"] for s in ml_segments])
            y_hat = self.model.predict(X)
            for seg, cls in zip(ml_segments, y_hat):
                pred[seg["indices"]] = int(cls)

        return clean_pts, pred

    def predict_hybrid(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trust geometry for ceiling/floor/wall; learn the rest. Returns (points_used, FULL13)."""
        return self._predict(points, frozenset(_STRUCT_IDX))

    def predict_hybrid_v2(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Trust geometry ONLY for floor/ceiling (unambiguous horizontal structure); let the
        model disambiguate every vertical plane (wall vs door vs window vs board). Motivated by
        the observation that geometry's blanket 'wall' call cannibalises coplanar doors/windows."""
        return self._predict(points, frozenset({"ceiling", "floor"}))

    def predict_feature_ml_only(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Every geometry segment classified by the model. Returns (points_used, FULL13 int)."""
        return self._predict(points, frozenset())
