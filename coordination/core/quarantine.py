from datetime import datetime, timedelta, timezone
from typing import Optional
from coordination.config import settings


def compute_quarantine_duration(old_multiplier: float, new_multiplier: float) -> float:
    """
    Q_duration = Q_base × log2(M_new / M_old)
    Returns duration in hours. Returns 0 for downgrades or equal multipliers.
    """
    return settings.quarantine_duration_hours(old_multiplier, new_multiplier)


def compute_quarantine_end(old_multiplier: float, new_multiplier: float) -> Optional[datetime]:
    """
    Returns the datetime when quarantine ends, or None if no quarantine needed.
    """
    duration_hours = compute_quarantine_duration(old_multiplier, new_multiplier)
    if duration_hours <= 0:
        return None
    return datetime.now(timezone.utc) + timedelta(hours=duration_hours)


def is_in_quarantine(quarantine_until: Optional[datetime]) -> bool:
    if quarantine_until is None:
        return False
    return datetime.now(timezone.utc) < quarantine_until


def quarantine_summary(old_multiplier: float, new_multiplier: float) -> dict:
    """
    Returns a human-readable summary of the quarantine decision.
    """
    duration_hours = compute_quarantine_duration(old_multiplier, new_multiplier)
    quarantine_end = compute_quarantine_end(old_multiplier, new_multiplier)
    return {
        "quarantine_required":  duration_hours > 0,
        "duration_hours":       round(duration_hours, 2),
        "quarantine_until":     quarantine_end.isoformat() if quarantine_end else None,
        "old_multiplier":       old_multiplier,
        "new_multiplier":       new_multiplier,
        "multiplier_ratio":     round(new_multiplier / old_multiplier, 4)
                                if old_multiplier > 0 else None,
    }
