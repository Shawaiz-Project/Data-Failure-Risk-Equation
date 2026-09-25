"""Trend detection over a series of DFRE risk scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class TrendReport:
    count: int
    slope: float               # risk units per batch (least squares)
    direction: str             # "increasing" | "decreasing" | "stable"
    latest: Optional[float]
    ewma: Optional[float]
    cusum: float               # cumulative upward deviation from the mean
    alarm: bool                # CUSUM breach or clearly rising slope
    forecast: List[float]      # naive linear forecast


class TrendDetector:
    """Detect rising data-failure risk before it breaches the policy.

    Parameters
    ----------
    window : int, default 20
        Only the most recent ``window`` points are analyzed.
    ewma_alpha : float, default 0.3
        Smoothing factor for the exponentially weighted moving average.
    cusum_threshold : float, default 0.5
        Alarm when cumulative upward deviation (in risk units) exceeds this.
    slope_threshold : float, default 0.005
        Minimum |slope| (risk per batch) considered a real trend.
    forecast_steps : int, default 3
        Length of the naive linear forecast.

    Examples
    --------
    >>> from dfre.monitoring.trends import TrendDetector
    >>> TrendDetector().analyze([0.1] * 10).direction
    'stable'
    """

    def __init__(
        self,
        window: int = 20,
        ewma_alpha: float = 0.3,
        cusum_threshold: float = 0.5,
        slope_threshold: float = 0.005,
        forecast_steps: int = 3,
    ) -> None:
        self.window = int(window)
        self.ewma_alpha = float(ewma_alpha)
        self.cusum_threshold = float(cusum_threshold)
        self.slope_threshold = float(slope_threshold)
        self.forecast_steps = int(forecast_steps)

    def analyze(self, risks: Sequence[float]) -> TrendReport:
        series = [float(r) for r in risks][-self.window:]
        if not series:
            return TrendReport(0, 0.0, "stable", None, None, 0.0, False, [])

        arr = np.asarray(series)
        n = len(arr)

        if n >= 2:
            x = np.arange(n, dtype=float)
            slope = float(np.polyfit(x, arr, 1)[0])
        else:
            slope = 0.0

        ewma = float(arr[0])
        for v in arr[1:]:
            ewma = self.ewma_alpha * float(v) + (1 - self.ewma_alpha) * ewma

        mean = float(arr.mean())
        cusum = 0.0
        cusum_max = 0.0
        for v in arr:
            cusum = max(0.0, cusum + (float(v) - mean))
            cusum_max = max(cusum_max, cusum)

        if slope > self.slope_threshold:
            direction = "increasing"
        elif slope < -self.slope_threshold:
            direction = "decreasing"
        else:
            direction = "stable"

        alarm = cusum_max >= self.cusum_threshold or (
            direction == "increasing" and float(arr[-1]) > mean
        )

        last_x = float(n - 1)
        intercept = float(arr.mean() - slope * x.mean()) if n >= 2 else float(arr[-1])
        forecast = [
            float(np.clip(intercept + slope * (last_x + k), 0.0, 1.0))
            for k in range(1, self.forecast_steps + 1)
        ] if n >= 2 else [float(arr[-1])] * self.forecast_steps

        return TrendReport(
            count=n,
            slope=slope,
            direction=direction,
            latest=float(arr[-1]),
            ewma=ewma,
            cusum=float(cusum_max),
            alarm=bool(alarm),
            forecast=forecast,
        )
