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

    def _predict(self, points: np.ndarray, trust: frozenset,
                 geo: dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict FULL13 per-point labels.

        ``trust`` is the set of geometry structural labels taken directly from the geometry
        rules; every other segment is classified by the model. An empty set == feature-ML-only.

        ``geo`` optionally supplies a precomputed ``run_geometry`` result for ``points`` so
        callers that already ran geometry (e.g. the label-efficiency sweep, which evaluates
        the same rooms across many N) can avoid recomputing it. Geometry is deterministic
        (seeded RANSAC/DBSCAN), so passing the cached ``geo`` is byte-identical to recomputing.
        """
        geo = geo if geo is not None else self.predictor.run_geometry(points)
        return self._label_from_segments(geo["clean_pts"], self.predictor.segments(geo), trust)

    def _label_from_segments(self, clean_pts: np.ndarray, segs: list,
                             trust: frozenset) -> Tuple[np.ndarray, np.ndarray]:
        """Core hybrid rule: take geometry's structural label directly for any segment whose
        label is in ``trust``; classify every other segment with the model.

        Operates purely on precomputed ``clean_pts`` + segment records, so callers that have
        already run geometry can drop the heavy geo objects (cKDTree, Open3D plane/cluster
        clouds) and keep only these lightweight arrays — critical for memory-bounded sweeps.
        """
        pred = np.full(len(clean_pts), _CLUTTER, dtype=np.int64)

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

    def predict_hybrid_from_segments(self, clean_pts: np.ndarray,
                                     segs: list) -> Tuple[np.ndarray, np.ndarray]:
        """Hybrid prediction (trust geometry for ceiling/floor/wall) from precomputed
        ``clean_pts`` + ``segs``, with no heavy geo held in memory. Identical result to
        ``predict_hybrid`` since segments are a deterministic function of the geometry."""
        return self._label_from_segments(clean_pts, segs, frozenset(_STRUCT_IDX))

    def predict_hybrid(self, points: np.ndarray,
                       geo: dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Trust geometry for ceiling/floor/wall; learn the rest. Returns (points_used, FULL13)."""
        return self._predict(points, frozenset(_STRUCT_IDX), geo=geo)

    def predict_hybrid_v2(self, points: np.ndarray,
                          geo: dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Trust geometry ONLY for floor/ceiling (unambiguous horizontal structure); let the
        model disambiguate every vertical plane (wall vs door vs window vs board). Motivated by
        the observation that geometry's blanket 'wall' call cannibalises coplanar doors/windows."""
        return self._predict(points, frozenset({"ceiling", "floor"}), geo=geo)

    def predict_feature_ml_only(self, points: np.ndarray,
                                geo: dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Every geometry segment classified by the model. Returns (points_used, FULL13 int)."""
        return self._predict(points, frozenset(), geo=geo)
