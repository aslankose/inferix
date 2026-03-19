from coordination.models.node import HardwareClass, HardwareProfile
from coordination.config import settings


# Multiplier tier boundaries (API score thresholds)
TIER_THRESHOLDS = [
    (0.00, HardwareClass.MOBILE_EDGE),
    (0.05, HardwareClass.CPU_ONLY),
    (0.15, HardwareClass.ENTRY_CONSUMER_GPU),
    (0.35, HardwareClass.MID_CONSUMER_GPU),
    (0.55, HardwareClass.HIGH_CONSUMER_GPU),
    (0.70, HardwareClass.PROSUMER_GPU),
    (0.82, HardwareClass.PROFESSIONAL_ACCEL),
    (0.92, HardwareClass.DATACENTER_ACCEL),
]

MULTIPLIER_MAP = {
    HardwareClass.MOBILE_EDGE:          settings.MULTIPLIER_MOBILE_EDGE,
    HardwareClass.CPU_ONLY:             settings.MULTIPLIER_CPU_ONLY,
    HardwareClass.ENTRY_CONSUMER_GPU:   settings.MULTIPLIER_ENTRY_CONSUMER_GPU,
    HardwareClass.MID_CONSUMER_GPU:     settings.MULTIPLIER_MID_CONSUMER_GPU,
    HardwareClass.HIGH_CONSUMER_GPU:    settings.MULTIPLIER_HIGH_CONSUMER_GPU,
    HardwareClass.PROSUMER_GPU:         settings.MULTIPLIER_PROSUMER_GPU,
    HardwareClass.PROFESSIONAL_ACCEL:   settings.MULTIPLIER_PROFESSIONAL_ACCEL,
    HardwareClass.DATACENTER_ACCEL:     settings.MULTIPLIER_DATACENTER_ACCEL,
}


def compute_api_score(profile: HardwareProfile) -> float:
    """
    Compute the composite AI Performance Index from benchmark scores.
    API = 0.5 × S_matmul + 0.3 × S_memory + 0.2 × S_latency
    Returns 0.0 if benchmark scores are not yet available.
    """
    if any(s is None for s in [profile.matmul_score,
                                profile.memory_score,
                                profile.latency_score]):
        return 0.0

    return (
        settings.API_WEIGHT_MATMUL  * profile.matmul_score +
        settings.API_WEIGHT_MEMORY  * profile.memory_score +
        settings.API_WEIGHT_LATENCY * profile.latency_score
    )


def assign_hardware_class(api_score: float) -> HardwareClass:
    """
    Map API score to hardware class tier.
    Uses linear interpolation at tier boundaries to avoid cliff effects.
    """
    assigned_class = HardwareClass.MOBILE_EDGE
    for threshold, hw_class in TIER_THRESHOLDS:
        if api_score >= threshold:
            assigned_class = hw_class
    return assigned_class


def interpolate_multiplier(api_score: float) -> float:
    """
    Compute a linearly interpolated multiplier between tier boundaries.
    This prevents sharp jumps at tier boundaries.
    """
    hw_class = assign_hardware_class(api_score)
    base_multiplier = MULTIPLIER_MAP[hw_class]

    # Find the tier above (if any) for interpolation
    tier_list = list(TIER_THRESHOLDS)
    current_index = next(
        (i for i, (_, cls) in enumerate(tier_list) if cls == hw_class),
        len(tier_list) - 1
    )

    if current_index >= len(tier_list) - 1:
        return base_multiplier

    current_threshold = tier_list[current_index][0]
    next_threshold    = tier_list[current_index + 1][0]
    next_class        = tier_list[current_index + 1][1]
    next_multiplier   = MULTIPLIER_MAP[next_class]

    # Linear interpolation within the tier band
    band_width    = next_threshold - current_threshold
    band_position = (api_score - current_threshold) / band_width if band_width > 0 else 0
    interpolated  = base_multiplier + band_position * (next_multiplier - base_multiplier)

    return round(interpolated, 4)


def assign_multiplier(profile: HardwareProfile) -> tuple[HardwareClass, float]:
    """
    Main entry point. Returns (hardware_class, multiplier) for a given profile.
    If benchmark scores are not yet available, returns lowest tier.
    """
    api_score = compute_api_score(profile)
    profile.api_score = api_score
    hw_class   = assign_hardware_class(api_score)
    multiplier = interpolate_multiplier(api_score)
    return hw_class, multiplier
