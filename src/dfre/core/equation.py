"""Core DFRE equation, its decomposition, and the generalized N-signal form.

Classic (4 signals, cyclic pairwise terms):

    𝓡 = 1 − exp( −[ M + V + D + U + λ(MV+VD+DU+UM) + γ·MVDU ] / (1 + √N) )

Generalized (any named signals, per-signal weights, all pairs):

    𝓡 = 1 − exp( −[ Σᵢ wᵢsᵢ + λ·Σᵢ<ⱼ wᵢwⱼsᵢsⱼ + γ·Πᵢ sᵢ ] / (1 + √N) )
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Dict, Mapping, Optional, Union

import numpy as np

from .errors import InvalidInputError, InvalidParameterError


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_unit(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidInputError(f"{name} must be a real number, got {type(value).__name__}")
    if not 0.0 <= float(value) <= 1.0:
        raise InvalidInputError(f"{name} must be in [0, 1], got {value}")


def _validate_params(n: int, lam: float, gamma: float) -> None:
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise InvalidParameterError(f"n must be a positive integer, got {n}")
    if lam < 0:
        raise InvalidParameterError(f"lambda must be >= 0, got {lam}")
    if gamma < 0:
        raise InvalidParameterError(f"gamma must be >= 0, got {gamma}")


# ---------------------------------------------------------------------------
# Classic 4-signal equation (backward compatible with dfre 0.1.x)
# ---------------------------------------------------------------------------

def dfre(
    missing: float,
    invalid: float,
    drift: float,
    uncertainty: float,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
    *,
    return_components: bool = False,
) -> Union[float, Dict[str, float]]:
    """Compute DFRE risk for a single batch.

    Parameters
    ----------
    missing, invalid, drift, uncertainty : float
        Normalized signals in [0, 1].
    n : int, default 1000
        Batch size (number of observations).
    lam : float, default 1.0
        Pairwise interaction strength (λ ≥ 0).
    gamma : float, default 1.0
        Four-way interaction strength (γ ≥ 0).
    return_components : bool, default False
        If True, return the full decomposition as a dict.

    Examples
    --------
    >>> from dfre.core.equation import dfre
    >>> round(dfre(0.08, 0.04, 0.12, 0.20), 4)
    0.015
    """
    _validate_unit("missing", missing)
    _validate_unit("invalid", invalid)
    _validate_unit("drift", drift)
    _validate_unit("uncertainty", uncertainty)
    _validate_params(n, lam, gamma)

    M, V, D, U = float(missing), float(invalid), float(drift), float(uncertainty)

    main = M + V + D + U
    pairwise = lam * (M * V + V * D + D * U + U * M)
    four_way = gamma * (M * V * D * U)
    intensity = (main + pairwise + four_way) / (1.0 + math.sqrt(n))
    risk = 1.0 - math.exp(-intensity)

    if return_components:
        return {
            "risk": risk,
            "intensity": intensity,
            "main": main,
            "pairwise": pairwise,
            "four_way": four_way,
        }
    return risk


def dfre_array(
    missing: np.ndarray,
    invalid: np.ndarray,
    drift: np.ndarray,
    uncertainty: np.ndarray,
    n: Union[int, np.ndarray] = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Vectorized DFRE over arrays of signals (same shape)."""
    M = np.asarray(missing, dtype=float)
    V = np.asarray(invalid, dtype=float)
    D = np.asarray(drift, dtype=float)
    U = np.asarray(uncertainty, dtype=float)
    N = np.asarray(n, dtype=float)

    for name, arr in (("missing", M), ("invalid", V), ("drift", D), ("uncertainty", U)):
        if np.any((arr < 0) | (arr > 1)):
            raise InvalidInputError(f"{name} contains values outside [0, 1]")
    if np.any(N <= 0):
        raise InvalidParameterError("n must be > 0")
    if lam < 0 or gamma < 0:
        raise InvalidParameterError("lam and gamma must be >= 0")

    main = M + V + D + U
    pairwise = lam * (M * V + V * D + D * U + U * M)
    four_way = gamma * M * V * D * U
    intensity = (main + pairwise + four_way) / (1.0 + np.sqrt(N))
    return 1.0 - np.exp(-intensity)


# ---------------------------------------------------------------------------
# Generalized N-signal equation (new in 0.2.0)
# ---------------------------------------------------------------------------

def dfre_generalized(
    signals: Mapping[str, float],
    *,
    weights: Optional[Mapping[str, float]] = None,
    n: int = 1000,
    lam: float = 1.0,
    gamma: float = 1.0,
    normalize_by_n: bool = True,
    return_components: bool = False,
) -> Union[float, Dict[str, object]]:
    """DFRE over an arbitrary set of named signals.

    Parameters
    ----------
    signals : mapping of str → float
        Any number of normalized signals in [0, 1], e.g.
        ``{"M": 0.1, "V": 0.2, "freshness": 0.4}``.
    weights : mapping of str → float, optional
        Per-signal importance weights (default 1.0 each). Must be ≥ 0.
    n : int
        Batch size used for the (1 + √N) confidence normalization.
    lam, gamma : float
        Interaction strengths for the pairwise sum and the full product term.
    normalize_by_n : bool, default True
        Set False to skip the (1 + √N) division (raw intensity scoring).
    return_components : bool

    Notes
    -----
    Unlike the classic form (which uses the four *cyclic* pairs MV, VD, DU, UM),
    the generalized form sums over **all** C(k, 2) pairs.

    Examples
    --------
    >>> from dfre.core.equation import dfre_generalized
    >>> r = dfre_generalized({"M": 0.2, "V": 0.2}, n=100)
    >>> 0 <= r < 1
    True
    """
    if not signals:
        raise InvalidInputError("signals must contain at least one entry")

    _validate_params(n, lam, gamma)
    w: Dict[str, float] = {}
    for name, value in signals.items():
        _validate_unit(name, float(value))
        weight = 1.0 if weights is None else float(weights.get(name, 1.0))
        if weight < 0:
            raise InvalidParameterError(f"weight for {name!r} must be >= 0, got {weight}")
        w[name] = weight

    names = list(signals)
    main = sum(w[k] * float(signals[k]) for k in names)
    pairwise = lam * sum(
        w[a] * w[b] * float(signals[a]) * float(signals[b])
        for a, b in combinations(names, 2)
    )
    product = 1.0
    for k in names:
        product *= float(signals[k])
    higher = gamma * product if len(names) >= 2 else 0.0

    denom = (1.0 + math.sqrt(n)) if normalize_by_n else 1.0
    intensity = (main + pairwise + higher) / denom
    risk = 1.0 - math.exp(-intensity)

    if return_components:
        return {
            "risk": risk,
            "intensity": intensity,
            "main": main,
            "pairwise": pairwise,
            "higher_order": higher,
            "signals": {k: float(v) for k, v in signals.items()},
            "weights": w,
        }
    return risk


# ---------------------------------------------------------------------------
# Inverse and budget helpers (new in 0.2.0)
# ---------------------------------------------------------------------------

def inverse_dfre(risk: float, n: int = 1000) -> float:
    """Return the total (unnormalized) signal budget that produces ``risk``.

    Useful for capacity planning: "how much combined signal can we tolerate
    before breaching the HIGH band?"
    """
    if not 0.0 <= risk < 1.0:
        raise InvalidInputError(f"risk must be in [0, 1), got {risk}")
    _validate_params(n, 0.0, 0.0)
    intensity = -math.log(1.0 - risk)
    return intensity * (1.0 + math.sqrt(n))
