"""Threshold calibration via cost-weighted search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..policy.bands import Policy


@dataclass
class ThresholdResult:
    low_max: float
    moderate_max: float
    high_max: float
    total_cost: float
    curve: list


def calibrate_thresholds(
    y_true,
    y_score,
    *,
    cost_fp: float = 1.0,
    cost_fn: float = 10.0,
    grid: int = 51,
) -> ThresholdResult:
    """Select three thresholds minimizing expected cost.

    Places LOW/MODERATE/HIGH bounds so that alerts at the HIGH or CRITICAL
    band minimize ``cost_fp·FP + cost_fn·FN``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    thresholds = np.linspace(0.0, 1.0, grid)
    curve = []

    for t in thresholds:
        pred = (y_score >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        curve.append({"threshold": float(t), "cost": cost_fp * fp + cost_fn * fn,
                      "fp": fp, "fn": fn})

    best = min(curve, key=lambda r: r["cost"])
    best_t = best["threshold"]

    low_max = float(np.clip(best_t * 0.5, 0.05, 0.95))
    moderate_max = float(np.clip(best_t, low_max + 0.01, 0.98))
    high_max = float(np.clip(best_t * 1.5, moderate_max + 0.01, 0.99))

    return ThresholdResult(
        low_max=low_max,
        moderate_max=moderate_max,
        high_max=high_max,
        total_cost=float(best["cost"]),
        curve=curve,
    )


def to_policy(result: ThresholdResult) -> Policy:
    """Convert a ThresholdResult into a Policy."""
    return Policy(
        low_max=result.low_max,
        moderate_max=result.moderate_max,
        high_max=result.high_max,
    )
