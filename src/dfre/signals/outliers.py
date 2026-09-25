"""Outlier signal (O) — fraction of anomalous rows."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .base import Signal


class OutlierSignal(Signal):
    """Fraction of rows flagged as outliers across selected numeric columns.

    A row is an outlier if ANY of its watched columns is extreme.

    Parameters
    ----------
    columns : list of str, optional
        Columns to inspect (None = all numeric columns of a DataFrame).
    method : {"iqr", "zscore"}, default "iqr"
    threshold : float, optional
        IQR multiplier (default 1.5) or |z| cutoff (default 3.0).

    Examples
    --------
    >>> import numpy as np
    >>> from dfre.signals.outliers import OutlierSignal
    >>> x = np.array([1.0, 1.1, 0.9, 1.0, 50.0])
    >>> round(OutlierSignal().compute(x.reshape(-1, 1)), 1)
    0.2
    """

    name = "outliers"

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: Optional[float] = None,
    ) -> None:
        if method not in {"iqr", "zscore"}:
            raise ValueError(f"method must be iqr/zscore, got {method}")
        self.columns = list(columns) if columns else None
        self.method = method
        self.threshold = float(threshold) if threshold is not None else (
            1.5 if method == "iqr" else 3.0
        )

    def compute(self, data) -> float:
        matrix = self._to_matrix(data)
        if matrix.size == 0:
            return 0.0
        n = matrix.shape[0]
        outlier = np.zeros(n, dtype=bool)
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            col = col[~np.isnan(col)]
            if col.size < 4:
                continue
            outlier |= self._flags(matrix[:, j], col)
        return float(outlier.mean())

    def _flags(self, full: np.ndarray, clean: np.ndarray) -> np.ndarray:
        if self.method == "iqr":
            q1, q3 = np.percentile(clean, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - self.threshold * iqr, q3 + self.threshold * iqr
            return (full < lo) | (full > hi)
        mu, sigma = float(clean.mean()), float(clean.std())
        if sigma == 0:
            return np.zeros(full.shape[0], dtype=bool)
        return np.abs((full - mu) / sigma) > self.threshold

    def _to_matrix(self, data) -> np.ndarray:
        if hasattr(data, "select_dtypes"):  # pandas
            df = data
            if self.columns:
                df = df[[c for c in self.columns if c in df.columns]]
            else:
                df = df.select_dtypes(include=[np.number])
            return df.to_numpy(dtype=float)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr


def outlier_rate(data, method: str = "iqr") -> float:
    """Functional shortcut for :class:`OutlierSignal`."""
    return OutlierSignal(method=method).compute(data)
