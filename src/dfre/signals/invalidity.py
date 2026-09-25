"""Invalidity signal (V)."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .base import Signal


RuleFn = Callable[[object], object]


class InvaliditySignal(Signal):
    """Fraction of rows failing validation rules.

    Parameters
    ----------
    rules : dict of str → callable
        Mapping of column name to a rule function. The function receives a
        column (pandas Series or numpy array) and returns a boolean mask where
        True = valid.
    """

    name = "invalidity"

    def __init__(self, rules: Dict[str, RuleFn]) -> None:
        self.rules = dict(rules)

    def compute(self, data) -> float:
        if not self.rules:
            return 0.0

        n = self._row_count(data)
        if n == 0:
            return 0.0

        valid = np.ones(n, dtype=bool)
        for col, rule in self.rules.items():
            column = self._get_column(data, col)
            try:
                mask = np.asarray(rule(column), dtype=bool).ravel()
            except Exception:
                mask = np.zeros(n, dtype=bool)
            if mask.size != n:
                mask = np.resize(mask, n)
            valid &= mask

        return float((~valid).mean())

    @staticmethod
    def _row_count(data) -> int:
        if hasattr(data, "shape"):
            return int(data.shape[0]) if len(data.shape) > 0 else 0
        try:
            return len(data)
        except TypeError:
            return 0

    @staticmethod
    def _get_column(data, col: str):
        if hasattr(data, "__getitem__"):
            try:
                return data[col]
            except Exception:
                pass
        raise KeyError(f"Column {col!r} not found in data")

    def failing_columns(self, data) -> Dict[str, float]:
        """Diagnostic: per-column failure rate."""
        out: Dict[str, float] = {}
        n = self._row_count(data)
        if n == 0:
            return out
        for col, rule in self.rules.items():
            try:
                column = self._get_column(data, col)
                mask = np.asarray(rule(column), dtype=bool).ravel()
                out[col] = float(1.0 - mask.mean())
            except Exception:
                out[col] = 1.0
        return out


def invalid_rate(df, rules: Dict[str, RuleFn]) -> float:
    """Functional shortcut for :class:`InvaliditySignal`."""
    return InvaliditySignal(rules).compute(df)
