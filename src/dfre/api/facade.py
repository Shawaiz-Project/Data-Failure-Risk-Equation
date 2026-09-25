"""Top-level DFRE facade that bundles calibration, policy, and history."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from ..core.types import DFREResult, Explanation
from ..policy.bands import Policy, DEFAULT_POLICY
from .one_liner import score, score_generalized, explain as _explain


class DFRE:
    """Stateful DFRE facade.

    Carries calibrated parameters (λ, γ), a policy, and an optional
    :class:`~dfre.monitoring.history.HistoryStore` across calls.

    Examples
    --------
    >>> from dfre import DFRE
    >>> model = DFRE(lam=1.0, gamma=1.0)
    >>> model.score(0.08, 0.04, 0.12, 0.20).band
    'LOW'
    """

    def __init__(
        self,
        *,
        lam: float = 1.0,
        gamma: float = 1.0,
        policy: Optional[Policy] = DEFAULT_POLICY,
        weights: Optional[Mapping[str, float]] = None,
        history_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.policy = policy
        self.weights = dict(weights or {})
        self.metadata = dict(metadata or {})
        self._history = None
        if history_path is not None:
            from ..monitoring.history import HistoryStore
            self._history = HistoryStore(history_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def score(
        self,
        missing: float,
        invalid: float,
        drift: float,
        uncertainty: float,
        *,
        n: int = 1000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DFREResult:
        merged = {**self.metadata, **(metadata or {})}
        result = score(
            missing, invalid, drift, uncertainty,
            n=n, lam=self.lam, gamma=self.gamma,
            policy=self.policy, metadata=merged,
        )
        self._persist(result)
        return result

    def score_generalized(
        self,
        signals: Mapping[str, float],
        *,
        n: int = 1000,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DFREResult:
        merged = {**self.metadata, **(metadata or {})}
        result = score_generalized(
            signals, weights=self.weights or None, n=n,
            lam=self.lam, gamma=self.gamma, policy=self.policy, metadata=merged,
        )
        self._persist(result)
        return result

    def explain(self, signals: Mapping[str, float], *, n: int = 1000) -> Explanation:
        """Attribute the risk to individual signals."""
        return _explain(signals, weights=self.weights or None,
                        n=n, lam=self.lam, gamma=self.gamma)

    def history(self):
        """The attached HistoryStore (None if history_path was not given)."""
        return self._history

    def _persist(self, result: DFREResult) -> None:
        if self._history is not None:
            try:
                self._history.append(result)
            except Exception:
                pass  # monitoring must never break scoring

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Persist parameters, weights and policy to a JSON file."""
        path = Path(path)
        path.write_text(
            json.dumps({
                "lam": self.lam,
                "gamma": self.gamma,
                "weights": self.weights,
                "policy": self.policy.to_dict() if self.policy else None,
                "metadata": self.metadata,
                "version": 2,
            }, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DFRE":
        """Load a previously saved DFRE instance (v1 and v2 files supported)."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = (
            Policy.from_dict(data["policy"]) if data.get("policy") else None
        )
        return cls(
            lam=data.get("lam", 1.0),
            gamma=data.get("gamma", 1.0),
            policy=policy,
            weights=data.get("weights"),
            metadata=data.get("metadata", {}),
        )
