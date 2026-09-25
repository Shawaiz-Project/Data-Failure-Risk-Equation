"""Missingness signal (M)."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .base import Signal


_DEFAULT_SENTINELS = (-1, -999, -9999, "", "NA", "N/A", "null", "NULL",
                      "None", "none", "unknown", "Unknown")


class MissingnessSignal(Signal):
    """Fraction of missing expected values in a batch.

    Supports pandas DataFrames, numpy arrays, and generic iterables.
    Treats configurable sentinel values as missing.
    """

    name = "missingness"

    def __init__(self, sentinels: Optional[Iterable] = None) -> None:
        self.sentinels = tuple(sentinels) if sentinels is not None else _DEFAULT_SENTINELS

    def compute(self, data) -> float:
        if data is None:
            return 0.0
        # pandas
        if hasattr(data, "isna") and hasattr(data, "size"):
            df = data.replace(list(self.sentinels), np.nan) if self.sentinels else data
            if df.size == 0:
                return 0.0
            return float(df.isna().sum().sum() / df.size)

        # numpy or generic
        arr = np.asarray(data, dtype=object)
        if arr.size == 0:
            return 0.0
        missing = np.array([v is None or (isinstance(v, float) and np.isnan(v))
                            or v in self.sentinels for v in arr.ravel()])
        return float(missing.mean())


def missing_rate(df) -> float:
    """Functional shortcut for :class:`MissingnessSignal`."""
    return MissingnessSignal().compute(df)
