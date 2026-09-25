"""Type definitions and lightweight containers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DFREResult:
    """Result of a single DFRE evaluation.

    Attributes
    ----------
    risk : float
        Aggregate risk 𝓡 ∈ [0, 1).
    intensity : float
        Unbounded risk intensity Z ≥ 0.
    main_component : float
        Weighted sum of the individual signals.
    pairwise_component : float
        λ-weighted interaction contribution.
    four_way_component : float
        γ-weighted higher-order product contribution.
    inputs : Dict[str, float]
        Original signal values plus N, λ, γ.
    band : Optional[str]
        Band label if a policy was attached.
    action : Optional[str]
        Recommended action if a policy was attached.
    extra_signals : Dict[str, float]
        Non-core signals (freshness, volume, ...) when generalized scoring is used.
    weights : Dict[str, float]
        Per-signal weights actually applied.
    metadata : Dict[str, Any]
        Free-form metadata (batch id, model version, etc.).
    """

    risk: float
    intensity: float
    main_component: float
    pairwise_component: float
    four_way_component: float
    inputs: Dict[str, float] = field(default_factory=dict)
    band: Optional[str] = None
    action: Optional[str] = None
    extra_signals: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        band = f" band={self.band}" if self.band else ""
        return f"DFREResult(risk={self.risk:.4f}{band})"


@dataclass(frozen=True)
class SignalSet:
    """Container for the four normalized core signals."""

    M: float
    V: float
    D: float
    U: float
    N: int = 1000

    def as_dict(self) -> Dict[str, float]:
        return {"M": self.M, "V": self.V, "D": self.D, "U": self.U, "N": float(self.N)}


@dataclass(frozen=True)
class Explanation:
    """Per-signal attribution of a DFRE score.

    contributions : leave-one-out drop in risk when each signal is zeroed.
    shares        : contributions normalized to sum to 1 (0 when risk is 0).
    top_driver    : name of the signal with the largest contribution.
    text          : human-readable one-paragraph explanation.
    """

    contributions: Dict[str, float]
    shares: Dict[str, float]
    top_driver: Optional[str]
    text: str

    def ranked(self) -> List[str]:
        """Signal names sorted by contribution, descending."""
        return sorted(self.contributions, key=self.contributions.get, reverse=True)  # type: ignore[arg-type]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
