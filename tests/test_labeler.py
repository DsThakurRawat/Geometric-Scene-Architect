"""Tests for SemanticLabeler — 20 test cases."""
from src.semantic_labeler import SemanticLabeler, LABEL_COLORS
from tests.conftest import make_plane, make_cluster

SCENE_HEIGHT = 3.0

# ── Plane labeling ────────────────────────────────────────────────────────────

class TestLabelPlanes:
    def test_floor_at_bottom(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        planes = labeler.label_planes([make_plane([0, 0, 1], 0.01)], SCENE_HEIGHT)
        assert planes[0]["label"] == "floor"

    def test_ceiling_at_top(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        planes = labeler.label_planes([make_plane([0, 0, 1], 2.8)], SCENE_HEIGHT)
        assert planes[0]["label"] == "ceiling"

    def test_wall_x_facing(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        planes = labeler.label_planes([make_plane([1, 0, 0], 1.5)], SCENE_HEIGHT)
        assert planes[0]["label"] == "wall"

    def test_wall_y_facing(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        planes = labeler.label_planes([make_plane([0, 1, 0], 1.5)], SCENE_HEIGHT)
        assert planes[0]["label"] == "wall"

    def test_slanted_plane_is_unknown(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        # 45° diagonal — neither horizontal nor vertical
        planes = labeler.label_planes([make_plane([0.707, 0, 0.707], 1.0)], SCENE_HEIGHT)
        assert planes[0]["label"] == "unknown"

    def test_horizontal_mid_height_is_horizontal_surface(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        planes = labeler.label_planes([make_plane([0, 0, 1], 1.0)], SCENE_HEIGHT)
        assert planes[0]["label"] == "horizontal_surface"

    def test_label_is_always_set(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.707]]
        for n in normals:
            plane = make_plane(n, 1.0)
            result = labeler.label_planes([plane], SCENE_HEIGHT)
            assert "label" in result[0]

    def test_zero_normal_vector_gets_unknown(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        plane = make_plane([0, 0, 0], 1.0)
        plane["normal"] = [0.0, 0.0, 0.0]
        result = labeler.label_planes([plane], SCENE_HEIGHT)
        assert result[0]["label"] == "unknown"

    def test_degenerate_scene_height_zero(self, base_cfg):
        """Very small scene height should not crash the ceiling detection."""
        labeler = SemanticLabeler(base_cfg)
        planes = labeler.label_planes([make_plane([0, 0, 1], 0.5)], scene_height=0.0)
        assert "label" in planes[0]

    def test_color_assigned_to_inlier_cloud(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        plane = make_plane([0, 0, 1], 0.01)
        result = labeler.label_planes([plane], SCENE_HEIGHT)
        assert result[0]["inlier_cloud"].has_colors()

    def test_empty_planes_list(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        result = labeler.label_planes([], SCENE_HEIGHT)
        assert result == []


# ── Cluster labeling ─────────────────────────────────────────────────────────

class TestLabelClusters:
    def test_floor_level_tall_is_tall_furniture(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        cluster = make_cluster(0.0, 2.0, [1.0, 1.5, 2.0])
        result = labeler.label_clusters([cluster])
        assert result[0]["label"] == "tall_furniture"

    def test_floor_level_normal_is_furniture(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        cluster = make_cluster(0.0, 0.75, [1.5, 1.2, 0.75])
        result = labeler.label_clusters([cluster])
        assert result[0]["label"] == "furniture"

    def test_high_z_base_is_high_fixture(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        cluster = make_cluster(2.5, 2.9, [0.6, 0.6, 0.4])
        result = labeler.label_clusters([cluster])
        assert result[0]["label"] == "high_fixture"

    def test_mid_height_small_footprint_is_small_object(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        cluster = make_cluster(0.75, 1.1, [0.3, 0.3, 0.35])
        result = labeler.label_clusters([cluster])
        assert result[0]["label"] == "small_object"

    def test_label_key_always_set(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        scenarios = [
            (0.0, 0.5, [0.5, 0.5, 0.5]),   # floor level
            (0.5, 1.0, [0.3, 0.3, 0.5]),   # mid
            (2.0, 2.5, [0.5, 0.5, 0.5]),   # high
        ]
        for z0, z1, d in scenarios:
            cluster = make_cluster(z0, z1, d)
            result = labeler.label_clusters([cluster])
            assert "label" in result[0]

    def test_color_painted_on_cluster_cloud(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        cluster = make_cluster(0.0, 0.75, [1.0, 1.0, 0.75])
        result = labeler.label_clusters([cluster])
        assert result[0]["cloud"].has_colors()

    def test_empty_clusters_list(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        result = labeler.label_clusters([])
        assert result == []

    def test_all_labels_in_color_map(self, base_cfg):
        labeler = SemanticLabeler(base_cfg)
        scenarios = [
            (0.0, 2.2, [1.0, 1.5, 2.2]),   # tall_furniture
            (0.0, 0.75, [1.5, 1.2, 0.75]), # furniture
            (2.5, 2.9, [0.6, 0.6, 0.4]),   # high_fixture
            (0.75, 1.1, [0.3, 0.3, 0.35]), # small_object
        ]
        for z0, z1, d in scenarios:
            cluster = make_cluster(z0, z1, d)
            result = labeler.label_clusters([cluster])
            lbl = result[0]["label"]
            assert lbl in LABEL_COLORS, f"Label '{lbl}' not in COLOR_MAP"

    def test_label_colors_dict_has_all_expected_keys(self):
        expected = {"floor", "ceiling", "wall", "furniture", "tall_furniture",
                    "small_object", "high_fixture", "horizontal_surface", "unknown", "noise"}
        assert expected.issubset(set(LABEL_COLORS.keys()))
