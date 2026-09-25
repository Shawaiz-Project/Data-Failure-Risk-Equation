"""Explainability: leave-one-out attribution of a DFRE score.

For each signal sᵢ we zero it out, recompute the risk, and attribute the drop
to that signal. This correctly credits interaction terms (a signal involved in
strong pairwise products gets credited for them).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from .equation import dfre_generalized
from .types import Explanation


def explain(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> Explanation:
    """Attribute a generalized DFRE score to its individual signals.

    Examples
    --------
    >>> from dfre.core.explain import explain
    >>> e = explain({"M": 0.6, "V": 0.1, "D": 0.1, "U": 0.1})
    >>> e.top_driver
    'M'
    """
    base = float(dfre_generalized(signals, weights=weights, n=n, lam=lam, gamma=gamma))

    contributions: Dict[str, float] = {}
    for name in signals:
        reduced = {k: (0.0 if k == name else v) for k, v in signals.items()}
        r = float(dfre_generalized(reduced, weights=weights, n=n, lam=lam, gamma=gamma))
        contributions[name] = max(0.0, base - r)

    total = sum(contributions.values())
    if total > 0:
        shares = {k: v / total for k, v in contributions.items()}
        top = max(contributions, key=contributions.get)  # type: ignore[arg-type]
    else:
        shares = {k: 0.0 for k in contributions}
        top = None

    if top is None or base == 0.0:
        text = "All signals are zero — the batch carries no measurable failure risk."
    else:
        ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        parts = ", ".join(f"{k} {v:.0%}" for k, v in ranked if v > 0.01)
        text = (
            f"Risk 𝓡 = {base:.4f} is driven mainly by {top} "
            f"({shares[top]:.0%} of attributable risk). Breakdown: {parts}."
        )

    return Explanation(contributions=contributions, shares=shares, top_driver=top, text=text)
