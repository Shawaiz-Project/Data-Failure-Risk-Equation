"""Volume signal (L) — row-count anomalies between batches."""

from __future__ import annotations

import math

from .base import Signal


class VolumeSignal(Signal):
    """Detect unexpected batch sizes (dropped or duplicated data).

    Uses ``|log(current / expected)| / scale``, clipped to [0, 1]. A batch that
    is half or twice the expected size has |log ratio| ≈ 0.69, so with the
    default ``scale=1.0`` that maps to ~0.69.

    Parameters
    ----------
    expected_rows : int, optional
        Reference row count. May be omitted and passed to :meth:`compute`.
    scale : float, default 1.0
        Log-ratio considered "maximal".
    """

    name = "volume"

    def __init__(self, expected_rows: int = 0, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.expected_rows = int(expected_rows)
        self.scale = float(scale)

    def compute(self, current_rows: int, expected_rows: int = 0) -> float:
        expected = expected_rows or self.expected_rows
        current = int(current_rows)
        if expected <= 0:
            return 0.0
        if current <= 0:
            return 1.0  # empty batch = maximal volume anomaly
        ratio = abs(math.log(current / expected))
        return float(min(1.0, ratio / self.scale))


def volume_rate(current_rows: int, expected_rows: int, scale: float = 1.0) -> float:
    """Functional shortcut for :class:`VolumeSignal`."""
    return VolumeSignal(expected_rows, scale).compute(current_rows)
