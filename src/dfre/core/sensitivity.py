"""Sensitivity analysis for the generalized DFRE equation.

- ``partial_derivatives``: exact analytic ∂𝓡/∂sᵢ (which signal moves risk fastest).
- ``tornado``: risk swing when each signal moves ±δ — for tornado charts.
- ``elasticity``: % change in risk per 1% change in each signal.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Mapping, Optional

from .equation import dfre_generalized
from .errors import InvalidInputError


def partial_derivatives(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """Exact analytic gradient of 𝓡 with respect to each signal.

    With Z = intensity and w = weights:
        ∂𝓡/∂sᵢ = e^(−Z) · [ wᵢ + λ·Σⱼ≠ᵢ wᵢwⱼsⱼ + γ·Πⱼ≠ᵢ sⱼ ] / (1 + √N)
    """
    if not signals:
        raise InvalidInputError("signals must not be empty")
    names = list(signals)
    w = {k: (1.0 if weights is None else float(weights.get(k, 1.0))) for k in names}

    comps = dfre_generalized(
        signals, weights=weights, n=n, lam=lam, gamma=gamma, return_components=True
    )
    Z = float(comps["intensity"])  # type: ignore[index]
    denom = 1.0 + math.sqrt(n)
    factor = math.exp(-Z) / denom

    out: Dict[str, float] = {}
    for i in names:
        others = [j for j in names if j != i]
        d_pair = lam * sum(w[i] * w[j] * float(signals[j]) for j in others)
        prod = 1.0
        for j in others:
            prod *= float(signals[j])
        d_higher = gamma * prod if len(names) >= 2 else 0.0
        out[i] = factor * (w[i] + d_pair + d_higher)
    return out


def tornado(
    signals: Mapping[str, float],
    *,
    delta: float = 0.1,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """Risk at sᵢ±δ for each signal (others fixed). Sorted by swing, descending."""
    base = float(dfre_generalized(signals, weights=weights, n=n, lam=lam, gamma=gamma))
    out: Dict[str, Dict[str, float]] = {}
    for name, value in signals.items():
        lo = {**signals, name: max(0.0, float(value) - delta)}
        hi = {**signals, name: min(1.0, float(value) + delta)}
        r_lo = float(dfre_generalized(lo, weights=weights, n=n, lam=lam, gamma=gamma))
        r_hi = float(dfre_generalized(hi, weights=weights, n=n, lam=lam, gamma=gamma))
        out[name] = {
            "base": base,
            "risk_minus": r_lo,
            "risk_plus": r_hi,
            "swing": r_hi - r_lo,
        }
    return dict(sorted(out.items(), key=lambda kv: kv[1]["swing"], reverse=True))


def elasticity(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> Dict[str, float]:
    """(∂𝓡/∂sᵢ)·(sᵢ/𝓡): percentage change in risk per 1% signal change."""
    base = float(dfre_generalized(signals, weights=weights, n=n, lam=lam, gamma=gamma))
    grads = partial_derivatives(signals, weights=weights, n=n, lam=lam, gamma=gamma)
    if base == 0:
        return {k: 0.0 for k in signals}
    return {k: grads[k] * float(signals[k]) / base for k in signals}
