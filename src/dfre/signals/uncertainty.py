"""Uncertainty signal (U)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Signal


class UncertaintySignal(Signal):
    """Normalized model uncertainty.

    Parameters
    ----------
    method : {"entropy", "margin", "mc_dropout"}, default "entropy"
        - "entropy": normalized predictive entropy from probabilities.
        - "margin": 1 − mean(top1 − top2 probability); 1 = total confusion.
        - "mc_dropout": standard deviation across MC samples.
    """

    name = "uncertainty"

    def __init__(self, method: str = "entropy") -> None:
        if method not in {"entropy", "margin", "mc_dropout"}:
            raise ValueError(f"method must be entropy/margin/mc_dropout, got {method}")
        self.method = method

    def compute(self, data: Any) -> float:
        if self.method == "entropy":
            return self._entropy(data)
        if self.method == "margin":
            return self._margin(data)
        return self._mc_dropout(data)

    @staticmethod
    def _entropy(probabilities: Any) -> float:
        p = np.asarray(probabilities, dtype=float)
        if p.size == 0:
            return 0.0
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum(axis=-1, keepdims=True)

        k = p.shape[-1]
        if k <= 1:
            return 0.0

        h = -(p * np.log(p)).sum(axis=-1)
        normalized = h / np.log(k)
        return float(np.clip(normalized.mean(), 0.0, 1.0))

    @staticmethod
    def _margin(probabilities: Any) -> float:
        p = np.asarray(probabilities, dtype=float)
        if p.size == 0:
            return 0.0
        if p.ndim != 2 or p.shape[1] < 2:
            return 0.0
        top2 = np.sort(p, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return float(np.clip(1.0 - margin.mean(), 0.0, 1.0))

    @staticmethod
    def _mc_dropout(predictions: Any) -> float:
        preds = np.asarray(predictions, dtype=float)
        if preds.ndim != 3:
            raise ValueError("mc_dropout expects shape (T, B, C)")
        std = preds.std(axis=0).mean(axis=-1)
        c = preds.shape[-1]
        scale = np.sqrt(c) if c > 1 else 1.0
        return float(np.clip(std.mean() / scale, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Functional helpers
# ---------------------------------------------------------------------------

def normalized_entropy(probabilities: Any) -> float:
    """Normalized mean predictive entropy ∈ [0, 1]."""
    return UncertaintySignal("entropy").compute(probabilities)


def mc_dropout_uncertainty(predictions: Any) -> float:
    """Normalized MC-dropout uncertainty ∈ [0, 1]."""
    return UncertaintySignal("mc_dropout").compute(predictions)
