"""Freshness signal (F) — data staleness relative to an SLA."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Union

from .base import Signal


TimestampLike = Union[str, datetime, int, float]


def _to_datetime(value: TimestampLike) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    text = str(value).replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class FreshnessSignal(Signal):
    """How stale the batch is, as a fraction of the agreed freshness SLA.

    ``F = min(1, age / sla)`` — 0 means "just produced", 1 means "at or beyond
    the SLA; the data should not be trusted for time-sensitive decisions".

    Parameters
    ----------
    sla_seconds : float
        Maximum acceptable age in seconds (e.g. 3600 for an hourly feed).

    Examples
    --------
    >>> from datetime import datetime, timedelta, timezone
    >>> from dfre.signals.freshness import FreshnessSignal
    >>> sig = FreshnessSignal(sla_seconds=3600)
    >>> now = datetime.now(timezone.utc)
    >>> sig.compute(now, now=now)
    0.0
    """

    name = "freshness"

    def __init__(self, sla_seconds: float) -> None:
        if sla_seconds <= 0:
            raise ValueError("sla_seconds must be > 0")
        self.sla_seconds = float(sla_seconds)

    def compute(
        self,
        last_updated: TimestampLike,
        *,
        now: Optional[TimestampLike] = None,
    ) -> float:
        ref = _to_datetime(last_updated)
        current = _to_datetime(now) if now is not None else datetime.now(timezone.utc)
        age = max(0.0, (current - ref).total_seconds())
        return float(min(1.0, age / self.sla_seconds))


def freshness_rate(last_updated: TimestampLike, sla_seconds: float) -> float:
    """Functional shortcut for :class:`FreshnessSignal`."""
    return FreshnessSignal(sla_seconds).compute(last_updated)
