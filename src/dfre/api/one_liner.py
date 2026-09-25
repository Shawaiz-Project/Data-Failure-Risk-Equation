"""Top-level one-liner API.

    import dfre
    result = dfre.score(missing=0.08, invalid=0.04, drift=0.12, uncertainty=0.20)
    print(result.risk)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..core.equation import dfre as _dfre, dfre_array as _dfre_array, dfre_generalized
from ..core.explain import explain as _explain_core
from ..core.types import DFREResult, Explanation
from ..policy.bands import Policy, DEFAULT_POLICY


def score(
    missing: float,
    invalid: float,
    drift: float,
    uncertainty: float,
    *,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
    policy: Optional[Policy] = DEFAULT_POLICY,
    metadata: Optional[Dict[str, Any]] = None,
) -> DFREResult:
    """Compute DFRE risk from the four core normalized signals.

    This is the primary entry point. It mirrors the equation exactly and
    attaches an operational band + recommended action when a policy is given.
    """
    components = _dfre(
        missing, invalid, drift, uncertainty,
        n=n, lam=lam, gamma=gamma, return_components=True,
    )

    band = action = None
    if policy is not None:
        band = policy.classify(components["risk"])
        action = policy.action_for(band)

    return DFREResult(
        risk=components["risk"],
        intensity=components["intensity"],
        main_component=components["main"],
        pairwise_component=components["pairwise"],
        four_way_component=components["four_way"],
        inputs={
            "M": float(missing), "V": float(invalid),
            "D": float(drift), "U": float(uncertainty),
            "N": float(n), "lam": float(lam), "gamma": float(gamma),
        },
        band=band,
        action=action,
        metadata=metadata or {},
    )


def score_generalized(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
    policy: Optional[Policy] = DEFAULT_POLICY,
    metadata: Optional[Dict[str, Any]] = None,
) -> DFREResult:
    """Compute DFRE risk from any set of named signals (0.2.0+).

    Extra signals (anything not in M/V/D/U) are stored on
    ``result.extra_signals``.
    """
    comps = dfre_generalized(
        signals, weights=weights, n=n, lam=lam, gamma=gamma, return_components=True
    )
    core = {k: float(signals.get(k, 0.0)) for k in ("M", "V", "D", "U")}
    extras = {k: float(v) for k, v in signals.items() if k not in core}

    band = action = None
    if policy is not None:
        band = policy.classify(float(comps["risk"]))  # type: ignore[arg-type]
        action = policy.action_for(band)

    return DFREResult(
        risk=float(comps["risk"]),            # type: ignore[arg-type]
        intensity=float(comps["intensity"]),  # type: ignore[arg-type]
        main_component=float(comps["main"]),  # type: ignore[arg-type]
        pairwise_component=float(comps["pairwise"]),     # type: ignore[arg-type]
        four_way_component=float(comps["higher_order"]), # type: ignore[arg-type]
        inputs={**core, "N": float(n), "lam": float(lam), "gamma": float(gamma)},
        band=band,
        action=action,
        extra_signals=extras,
        weights=dict(comps["weights"]),  # type: ignore[arg-type]
        metadata=metadata or {},
    )


def explain(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> Explanation:
    """One-liner attribution: which signals drive the risk score?"""
    return _explain_core(signals, weights=weights, n=n, lam=lam, gamma=gamma)


def score_array(
    missing,
    invalid,
    drift,
    uncertainty,
    *,
    n=1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Vectorized risk over arrays of signals."""
    return _dfre_array(missing, invalid, drift, uncertainty, n=n, lam=lam, gamma=gamma)
