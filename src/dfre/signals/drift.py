"""Drift signal (D)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Signal


# ---------------------------------------------------------------------------
# Raw statistics
# ---------------------------------------------------------------------------

def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index."""
    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0

    breaks = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(breaks) < 3:
        return 0.0

    ref_counts, _ = np.histogram(ref, bins=breaks)
    cur_counts, _ = np.histogram(cur, bins=breaks)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, 1.0)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, 1.0)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks(reference: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov–Smirnov two-sample statistic."""
    try:
        from scipy import stats
    except ImportError as e:  # pragma: no cover
        raise ImportError("scipy is required for ks(); pip install dfre[scipy]") from e

    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    return float(stats.ks_2samp(ref, cur).statistic)


def wasserstein(reference: np.ndarray, current: np.ndarray) -> float:
    """1-Wasserstein distance."""
    try:
        from scipy import stats
    except ImportError as e:  # pragma: no cover
        raise ImportError("scipy is required for wasserstein(); pip install dfre[scipy]") from e

    ref = _clean(reference)
    cur = _clean(current)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    return float(stats.wasserstein_distance(ref, cur))


def total_variation(reference, current) -> float:
    """Total variation for categorical features."""
    ref = _value_counts(reference)
    cur = _value_counts(current)
    idx = ref.index.union(cur.index)
    return float(0.5 * np.abs(ref.reindex(idx, fill_value=0)
                              - cur.reindex(idx, fill_value=0)).sum())


def jensen_shannon(reference, current) -> float:
    """Jensen–Shannon divergence for categorical features ∈ [0, ln 2]."""
    ref = _value_counts(reference)
    cur = _value_counts(current)
    idx = ref.index.union(cur.index)
    p = ref.reindex(idx, fill_value=0).to_numpy(dtype=float)
    q = cur.reindex(idx, fill_value=0).to_numpy(dtype=float)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


# ---------------------------------------------------------------------------
# Public signal class
# ---------------------------------------------------------------------------

class DriftSignal(Signal):
    """Normalized distribution drift across a batch.

    Aggregates per-column drift scores and maps them to [0, 1] using a
    calibrated scale.
    """

    name = "drift"

    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: Optional[List[str]] = None,
        statistic: str = "psi",
        scale: float = 0.25,
    ) -> None:
        if statistic not in {"psi", "ks", "wasserstein"}:
            raise ValueError(f"statistic must be psi/ks/wasserstein, got {statistic}")
        if scale <= 0:
            raise ValueError("scale must be > 0")
        self.continuous_cols = list(continuous_cols)
        self.categorical_cols = list(categorical_cols or [])
        self.statistic = statistic
        self.scale = float(scale)

    def compute(self, reference, current) -> float:
        raw = self.raw(reference, current)
        if not raw:
            return 0.0
        aggregate = float(np.mean(list(raw.values())))
        return float(min(1.0, max(0.0, aggregate / self.scale)))

    def raw(self, reference, current) -> Dict[str, float]:
        """Return per-column raw drift values (unnormalized)."""
        out: Dict[str, float] = {}
        for col in self.continuous_cols:
            if col not in reference or col not in current:
                continue
            ref = np.asarray(reference[col], dtype=float)
            cur = np.asarray(current[col], dtype=float)
            out[col] = _compute_continuous(ref, cur, self.statistic)

        for col in self.categorical_cols:
            if col not in reference or col not in current:
                continue
            out[col] = total_variation(reference[col], current[col])
        return out

    def explain(self, reference, current) -> Dict[str, Any]:
        """Return raw, normalized, and aggregate per column."""
        raw = self.raw(reference, current)
        if not raw:
            return {"raw": {}, "normalized": 0.0}
        aggregate = float(np.mean(list(raw.values())))
        normalized = float(min(1.0, max(0.0, aggregate / self.scale)))
        return {
            "raw": raw,
            "aggregate_raw": aggregate,
            "normalized": normalized,
            "scale": self.scale,
            "statistic": self.statistic,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[~np.isnan(a)]


def _value_counts(series):
    try:
        return series.value_counts(normalize=True)
    except AttributeError:
        import pandas as pd
        return pd.Series(series).value_counts(normalize=True)


def _compute_continuous(ref: np.ndarray, cur: np.ndarray, statistic: str) -> float:
    if statistic == "psi":
        return psi(ref, cur)
    if statistic == "ks":
        return ks(ref, cur)
    if statistic == "wasserstein":
        return wasserstein(ref, cur)
    raise ValueError(statistic)
