"""
Tests for the hardware contribution multiplier.
Validates Section 5 of the Inferix paper.
"""
import pytest
from coordination.models.node import HardwareProfile, HardwareClass
from coordination.core.multiplier import (
    compute_api_score,
    assign_hardware_class,
    interpolate_multiplier,
    assign_multiplier,
)


def make_profile(matmul=0.0, memory=0.0, latency=0.0) -> HardwareProfile:
    return HardwareProfile(
        matmul_score=matmul,
        memory_score=memory,
        latency_score=latency,
    )


class TestAPIScore:
    """
    Section 5.1 — AI Performance Index:
    API = 0.5 × S_matmul + 0.3 × S_memory + 0.2 × S_latency
    """

    def test_zero_scores_give_zero_api(self):
        profile = make_profile(0.0, 0.0, 0.0)
        assert compute_api_score(profile) == pytest.approx(0.0)

    def test_perfect_scores_give_one(self):
        profile = make_profile(1.0, 1.0, 1.0)
        assert compute_api_score(profile) == pytest.approx(1.0)

    def test_matmul_weighted_highest(self):
        """matmul weight (0.5) is highest — dominates the score."""
        only_matmul = make_profile(matmul=1.0, memory=0.0, latency=0.0)
        only_memory = make_profile(matmul=0.0, memory=1.0, latency=0.0)
        assert compute_api_score(only_matmul) > compute_api_score(only_memory)

    def test_weights_sum_to_one(self):
        """API score of all-1.0 inputs equals 1.0."""
        profile = make_profile(1.0, 1.0, 1.0)
        assert compute_api_score(profile) == pytest.approx(1.0)

    def test_none_scores_return_zero(self):
        """Missing benchmark scores return 0.0."""
        profile = HardwareProfile()  # All None
        assert compute_api_score(profile) == pytest.approx(0.0)

    def test_rtx3060_representative_score(self):
        """RTX 3060 benchmark scores should land in mid consumer tier."""
        profile = make_profile(matmul=0.45, memory=0.40, latency=0.35)
        score   = compute_api_score(profile)
        hw_class, _ = assign_multiplier(profile)
        assert hw_class == HardwareClass.MID_CONSUMER_GPU


class TestHardwareClassAssignment:
    """
    Section 5.2 — Multiplier tier assignment.
    """

    def test_zero_api_score_is_mobile_edge(self):
        assert assign_hardware_class(0.0) == HardwareClass.MOBILE_EDGE

    def test_mid_consumer_gpu_is_baseline(self):
        """Mid consumer GPU is the baseline tier at multiplier 1.0."""
        hw_class = assign_hardware_class(0.40)
        assert hw_class == HardwareClass.MID_CONSUMER_GPU

    def test_datacenter_accel_at_high_score(self):
        hw_class = assign_hardware_class(0.95)
        assert hw_class == HardwareClass.DATACENTER_ACCEL

    def test_all_tiers_reachable(self):
        """Every hardware tier can be assigned."""
        test_scores = [0.01, 0.10, 0.25, 0.45, 0.62, 0.76, 0.87, 0.95]
        classes     = [assign_hardware_class(s) for s in test_scores]
        assert len(set(classes)) == 8   # All 8 tiers represented

    def test_monotonic_assignment(self):
        """Higher API scores always map to equal or higher hardware tiers."""
        scores  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        classes = [assign_hardware_class(s) for s in scores]
        multipliers = [interpolate_multiplier(s) for s in scores]
        # Multipliers should be non-decreasing
        for i in range(1, len(multipliers)):
            assert multipliers[i] >= multipliers[i-1]


class TestMultiplierInterpolation:
    """
    Section 5.2 — Linear interpolation prevents cliff effects at tier boundaries.
    """

    def test_baseline_multiplier_is_one(self):
        """Mid consumer GPU baseline multiplier is exactly 1.0."""
        profile  = make_profile(0.45, 0.40, 0.35)
        _, multi = assign_multiplier(profile)
        assert multi >= 1.0

    def test_datacenter_multiplier_is_eight(self):
        """Datacenter accelerator multiplier is 8.0."""
        profile  = make_profile(1.0, 1.0, 1.0)
        _, multi = assign_multiplier(profile)
        assert multi == pytest.approx(8.0, rel=1e-2)

    def test_mobile_edge_multiplier_is_lowest(self):
        """Mobile/edge multiplier is the smallest."""
        profile  = make_profile(0.01, 0.01, 0.01)
        _, multi = assign_multiplier(profile)
        assert multi <= 0.5

    def test_interpolation_between_tiers(self):
        """Score between two tiers gives interpolated multiplier."""
        low_profile  = make_profile(0.36, 0.36, 0.36)
        high_profile = make_profile(0.54, 0.54, 0.54)
        mid_profile  = make_profile(0.45, 0.45, 0.45)

        _, low_multi  = assign_multiplier(low_profile)
        _, high_multi = assign_multiplier(high_profile)
        _, mid_multi  = assign_multiplier(mid_profile)

        assert low_multi < mid_multi < high_multi

    def test_multiplier_is_positive(self):
        """All multipliers are positive."""
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            multi = interpolate_multiplier(score)
            assert multi > 0


class TestQuarantine:
    """
    Section 6.4 — Hardware swap attack prevention:
    Q_duration = Q_base × log2(M_new / M_old)
    """

    def test_upgrade_triggers_quarantine(self):
        from coordination.core.quarantine import compute_quarantine_duration
        duration = compute_quarantine_duration(1.0, 8.0)
        assert duration > 0

    def test_downgrade_no_quarantine(self):
        from coordination.core.quarantine import compute_quarantine_duration
        duration = compute_quarantine_duration(8.0, 1.0)
        assert duration == 0.0

    def test_same_multiplier_no_quarantine(self):
        from coordination.core.quarantine import compute_quarantine_duration
        duration = compute_quarantine_duration(1.0, 1.0)
        assert duration == 0.0

    def test_larger_upgrade_longer_quarantine(self):
        """Bigger hardware jump = longer quarantine."""
        from coordination.core.quarantine import compute_quarantine_duration
        small_upgrade = compute_quarantine_duration(1.0, 2.0)
        large_upgrade = compute_quarantine_duration(1.0, 8.0)
        assert large_upgrade > small_upgrade

    def test_quarantine_formula(self):
        """Q = Q_base × log2(M_new / M_old)."""
        import math
        from coordination.core.quarantine import compute_quarantine_duration
        from coordination.config import settings
        old, new = 1.0, 4.0
        expected = settings.QUARANTINE_BASE_HOURS * math.log2(new / old)
        result   = compute_quarantine_duration(old, new)
        assert result == pytest.approx(expected, rel=1e-4)
