"""Tests for Pydantic models in src/models.py — 15 test cases."""
import pytest
from pydantic import ValidationError
from src.models import PlaneResult, ClusterResult, SegmentationReport


# ── PlaneResult ──────────────────────────────────────────────────────────────

class TestPlaneResult:
    def _valid(self, **kwargs):
        defaults = dict(
            plane_id=0, label="floor", inlier_count=500, centroid_z=0.01,
            normal=[0.0, 0.0, 1.0], plane_model=[0.0, 0.0, 1.0, 0.0]
        )
        defaults.update(kwargs)
        return PlaneResult(**defaults)

    def test_valid_floor(self):
        p = self._valid(label="floor")
        assert p.label == "floor"

    def test_valid_all_labels(self):
        for lbl in ["floor", "ceiling", "wall", "horizontal_surface", "unknown", "noise"]:
            p = self._valid(label=lbl)
            assert p.label == lbl

    def test_invalid_label_raises(self):
        with pytest.raises(ValidationError, match="recognised semantic label"):
            self._valid(label="sofa")

    def test_zero_normal_raises(self):
        with pytest.raises(ValidationError, match="zero vector"):
            self._valid(normal=[0.0, 0.0, 0.0])

    def test_negative_inlier_count_raises(self):
        with pytest.raises(ValidationError):
            self._valid(inlier_count=-1)

    def test_normal_must_be_length_3(self):
        with pytest.raises(ValidationError):
            self._valid(normal=[0.0, 0.0])

    def test_plane_model_must_be_length_4(self):
        with pytest.raises(ValidationError):
            self._valid(plane_model=[0.0, 0.0, 1.0])


# ── ClusterResult ─────────────────────────────────────────────────────────────

class TestClusterResult:
    def _valid(self, **kwargs):
        defaults = dict(
            cluster_id=0, label="furniture", n_points=200,
            centroid=[1.0, 1.0, 0.4], dims=[1.0, 1.0, 0.75],
            z_min=0.0, z_max=0.75, footprint_m2=1.0
        )
        defaults.update(kwargs)
        return ClusterResult(**defaults)

    def test_valid_furniture(self):
        c = self._valid()
        assert c.label == "furniture"

    def test_z_min_gt_z_max_raises(self):
        with pytest.raises(ValidationError, match="z_min"):
            self._valid(z_min=2.0, z_max=0.5)

    def test_negative_dims_raises(self):
        with pytest.raises(ValidationError, match="non-negative"):
            self._valid(dims=[-1.0, 1.0, 0.5])

    def test_invalid_cluster_label_raises(self):
        with pytest.raises(ValidationError, match="recognised semantic label"):
            self._valid(label="bookshelf")

    def test_obb_fields_optional(self):
        c = self._valid()
        assert c.obb_extent is None
        assert c.obb_rotation_deg is None

    def test_obb_fields_accepted_when_present(self):
        c = self._valid(obb_extent=[1.0, 1.0, 0.75], obb_rotation_deg=15.3)
        assert c.obb_extent == [1.0, 1.0, 0.75]

    def test_negative_footprint_allowed_as_zero(self):
        """Footprint is always >= 0 (validator on ge=0)."""
        with pytest.raises(ValidationError):
            self._valid(footprint_m2=-0.1)


# ── SegmentationReport ────────────────────────────────────────────────────────

class TestSegmentationReport:
    def _plane(self, label="floor"):
        return PlaneResult(
            plane_id=0, label=label, inlier_count=500, centroid_z=0.01,
            normal=[0.0, 0.0, 1.0], plane_model=[0.0, 0.0, 1.0, 0.0]
        )

    def _cluster(self, label="furniture"):
        return ClusterResult(
            cluster_id=0, label=label, n_points=200, centroid=[1.0, 1.0, 0.4],
            dims=[1.0, 1.0, 0.75], z_min=0.0, z_max=0.75, footprint_m2=1.0
        )

    def test_empty_report_valid(self):
        r = SegmentationReport()
        assert r.structural_planes == []
        assert r.objects == []

    def test_label_counts_correct(self):
        r = SegmentationReport(
            structural_planes=[self._plane("floor"), self._plane("wall"), self._plane("floor")],
            objects=[self._cluster("furniture"), self._cluster("tall_furniture")]
        )
        counts = r.plane_label_counts
        assert counts["floor"] == 2
        assert counts["wall"] == 1
        assert r.object_label_counts["furniture"] == 1
