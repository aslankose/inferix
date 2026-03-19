"""
Tests for the grid-aware task scheduler.
Validates Section 7.3 of the Inferix paper.
"""
import pytest
from unittest.mock import MagicMock
from coordination.core.scheduler import GridAwareScheduler, GridState


def make_node(node_id, multiplier=1.0, reliability=1.0, status="active"):
    node = MagicMock()
    node.id                = node_id
    node.multiplier        = multiplier
    node.reliability_factor= reliability
    node.status            = status
    return node


class TestGridScore:
    """
    Section 7.3 — Grid priority formula:
    P_grid = w1×(1−L) + w2×(1−C) + w3×R
    """

    def test_clean_grid_scores_high(self):
        """Low load, low carbon, high renewable = high grid score."""
        scheduler = GridAwareScheduler()
        scheduler.update_grid_state(GridState(
            region="CLEAN", load_factor=0.1,
            carbon_intensity=0.05, renewable_fraction=0.95
        ))
        score = scheduler.get_grid_score("CLEAN")
        assert score > 0.8

    def test_dirty_grid_scores_low(self):
        """High load, high carbon, low renewable = low grid score."""
        scheduler = GridAwareScheduler()
        scheduler.update_grid_state(GridState(
            region="DIRTY", load_factor=0.9,
            carbon_intensity=0.9, renewable_fraction=0.05
        ))
        score = scheduler.get_grid_score("DIRTY")
        assert score < 0.2

    def test_unknown_region_returns_neutral(self):
        """Unknown region returns neutral score of 0.5."""
        scheduler = GridAwareScheduler()
        score = scheduler.get_grid_score("XX")
        assert score == pytest.approx(0.5)

    def test_norway_scores_higher_than_texas(self):
        """Norway (high renewable) should score higher than Texas (high fossil)."""
        scheduler = GridAwareScheduler()
        norway_score = scheduler.get_grid_score("NO")
        texas_score  = scheduler.get_grid_score("US-TX")
        assert norway_score > texas_score

    def test_grid_score_bounded(self):
        """Grid score is always between 0 and 1."""
        scheduler = GridAwareScheduler()
        for region in ["NO", "FR", "GB", "US-TX", "SG", "JP"]:
            score = scheduler.get_grid_score(region)
            assert 0.0 <= score <= 1.0

    def test_weight_sum(self):
        """Weights w1+w2+w3 = 1.0."""
        from coordination.config import settings
        total = (settings.GRID_WEIGHT_LOAD +
                 settings.GRID_WEIGHT_CARBON +
                 settings.GRID_WEIGHT_RENEWABLE)
        assert total == pytest.approx(1.0)

    def test_grid_state_update(self):
        """Updating grid state changes the score."""
        scheduler = GridAwareScheduler()
        scheduler.update_grid_state(GridState(
            region="GB", load_factor=0.9,
            carbon_intensity=0.9, renewable_fraction=0.05
        ))
        low_score = scheduler.get_grid_score("GB")

        scheduler.update_grid_state(GridState(
            region="GB", load_factor=0.1,
            carbon_intensity=0.05, renewable_fraction=0.95
        ))
        high_score = scheduler.get_grid_score("GB")
        assert high_score > low_score


class TestNodeSelection:
    """
    Node selection — grid as tiebreaker, performance dominates.
    """

    def test_selects_from_candidates(self):
        """Scheduler returns one of the candidate nodes."""
        scheduler = GridAwareScheduler()
        nodes     = [make_node("node-1"), make_node("node-2")]
        selected  = scheduler.select_node(nodes)
        assert selected is not None
        assert selected in nodes

    def test_empty_candidates_returns_none(self):
        scheduler = GridAwareScheduler()
        assert scheduler.select_node([]) is None

    def test_prefers_higher_performance(self):
        """Higher multiplier node scores higher than lower for same region."""
        scheduler = GridAwareScheduler()
        scheduler.register_node_region("low",  "US-NY")
        scheduler.register_node_region("high", "US-NY")

        low_node  = make_node("low",  multiplier=1.0)
        high_node = make_node("high", multiplier=8.0)

        low_score  = scheduler._score_node(low_node).final_score
        high_score = scheduler._score_node(high_node).final_score
        assert high_score > low_score

    def test_prefers_cleaner_grid_for_equal_hardware(self):
        """For equal hardware, node in cleaner grid region is preferred."""
        scheduler = GridAwareScheduler()
        scheduler.register_node_region("node-no", "NO")    # Norway — cleanest
        scheduler.register_node_region("node-sg", "SG")    # Singapore — dirtiest

        norway_node    = make_node("node-no", multiplier=1.0, reliability=1.0)
        singapore_node = make_node("node-sg", multiplier=1.0, reliability=1.0)

        norway_score    = scheduler._score_node(norway_node).final_score
        singapore_score = scheduler._score_node(singapore_node).final_score
        assert norway_score > singapore_score

    def test_node_score_components_bounded(self):
        """All score components are between 0 and 1."""
        scheduler = GridAwareScheduler()
        scheduler.register_node_region("node-1", "GB")
        node  = make_node("node-1", multiplier=1.325, reliability=0.8)
        score = scheduler._score_node(node)

        assert 0.0 <= score.grid_score  <= 1.0
        assert 0.0 <= score.perf_score  <= 1.0
        assert 0.0 <= score.reliability <= 1.0
        assert 0.0 <= score.final_score <= 1.0

    def test_region_registration(self):
        """Registered region is used for grid scoring."""
        scheduler = GridAwareScheduler()
        scheduler.register_node_region("node-1", "NO")
        node  = make_node("node-1")
        score = scheduler._score_node(node)
        assert score.region == "NO"
        assert score.grid_score == pytest.approx(
            scheduler.get_grid_score("NO"), rel=1e-4
        )

    def test_fairness_guarantee(self):
        """Underserved nodes receive tasks regardless of score."""
        scheduler = GridAwareScheduler()
        scheduler.register_node_region("strong", "NO")
        scheduler.register_node_region("weak",   "SG")

        strong = make_node("strong", multiplier=8.0)
        weak   = make_node("weak",   multiplier=0.05)

        # Give strong node many allocations
        scheduler._alloc_counter["strong"] = 100
        scheduler._alloc_counter["weak"]   = 0

        # Weak node should be selected due to fairness guarantee
        selected = scheduler._apply_min_allocation_guarantee([strong, weak])
        assert selected is not None
        assert selected.id == "weak"
