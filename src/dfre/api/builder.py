"""Composable builder API.

    from dfre import DFREBuilder
    from dfre.signals import MissingnessSignal, InvaliditySignal

    pipeline = (
        DFREBuilder()
        .with_missingness(MissingnessSignal())
        .with_invalidity(InvaliditySignal(rules))
        .with_drift(DriftSignal(["age", "bmi"], scale=0.25))
        .with_uncertainty(UncertaintySignal())
        .with_parameters(lam=1.0, gamma=1.0)
        .build()
    )
    result = pipeline.evaluate(batch, reference, probabilities)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..core.equation import dfre as _dfre, dfre_generalized
from ..core.types import DFREResult
from ..policy.bands import Policy, DEFAULT_POLICY
from ..signals.base import Signal
from ..signals.missingness import MissingnessSignal
from ..signals.invalidity import InvaliditySignal
from ..signals.drift import DriftSignal
from ..signals.uncertainty import UncertaintySignal


@dataclass
class _PipelineConfig:
    missingness: Signal
    invalidity: Signal
    drift: Signal
    uncertainty: Signal
    lam: float
    gamma: float
    extra_signals: Dict[str, Signal] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)


class DFREPipeline:
    """Fully-configured DFRE pipeline. Created via :class:`DFREBuilder`."""

    def __init__(self, config: _PipelineConfig, policy: Optional[Policy]) -> None:
        self._config = config
        self._policy = policy

    def evaluate(
        self,
        batch,
        reference=None,
        probabilities=None,
        *,
        context: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DFREResult:
        """Compute a full DFRE result for a batch.

        Parameters
        ----------
        batch : DataFrame / array / iterable
        reference : optional reference data for the drift signal
        probabilities : optional model probabilities for the uncertainty signal
        context : mapping, optional
            Inputs for extra signals, keyed by the names given to
            ``with_extra_signal`` — e.g. ``{"last_updated": ..., "expected_rows": ...}``.
        """
        cfg = self._config

        M = cfg.missingness.compute(batch)
        V = cfg.invalidity.compute(batch)
        D = cfg.drift.compute(reference, batch) if reference is not None else 0.0
        U = cfg.uncertainty.compute(probabilities) if probabilities is not None else 0.0
        n = _row_count(batch)

        ctx = dict(context or {})
        extras: Dict[str, float] = {}
        for name, signal in cfg.extra_signals.items():
            args = ctx.get(name, ())
            if not isinstance(args, (tuple, list)):
                args = (args,) if args is not None else ()
            try:
                extras[name] = float(signal.compute(batch, *args))
            except TypeError:
                extras[name] = float(signal.compute(*args))

        if extras:
            signals = {"M": M, "V": V, "D": D, "U": U, **extras}
            comps = dfre_generalized(
                signals, weights=cfg.weights or None, n=n,
                lam=cfg.lam, gamma=cfg.gamma, return_components=True,
            )
            risk = float(comps["risk"])            # type: ignore[index]
            intensity = float(comps["intensity"])  # type: ignore[index]
            main = float(comps["main"])            # type: ignore[index]
            pairwise = float(comps["pairwise"])    # type: ignore[index]
            higher = float(comps["higher_order"])  # type: ignore[index]
        else:
            comps = _dfre(M, V, D, U, n=n, lam=cfg.lam, gamma=cfg.gamma,
                          return_components=True)
            risk, intensity = comps["risk"], comps["intensity"]
            main, pairwise, higher = comps["main"], comps["pairwise"], comps["four_way"]

        band = action = None
        if self._policy is not None:
            band = self._policy.classify(risk)
            action = self._policy.action_for(band)

        return DFREResult(
            risk=risk,
            intensity=intensity,
            main_component=main,
            pairwise_component=pairwise,
            four_way_component=higher,
            inputs={"M": M, "V": V, "D": D, "U": U, "N": float(n),
                    "lam": cfg.lam, "gamma": cfg.gamma},
            band=band,
            action=action,
            extra_signals=extras,
            weights=cfg.weights,
            metadata=metadata or {},
        )


class DFREBuilder:
    """Fluent builder for :class:`DFREPipeline`."""

    def __init__(self) -> None:
        self._missingness: Signal = MissingnessSignal()
        self._invalidity: Signal = InvaliditySignal({})
        self._drift: Signal = DriftSignal([], scale=0.25)
        self._uncertainty: Signal = UncertaintySignal()
        self._extra: Dict[str, Signal] = {}
        self._weights: Dict[str, float] = {}
        self._lam: float = 1.0
        self._gamma: float = 1.0
        self._policy: Optional[Policy] = DEFAULT_POLICY

    def with_missingness(self, signal: Signal) -> "DFREBuilder":
        self._missingness = signal
        return self

    def with_invalidity(self, signal: Signal) -> "DFREBuilder":
        self._invalidity = signal
        return self

    def with_drift(self, signal: Signal) -> "DFREBuilder":
        self._drift = signal
        return self

    def with_uncertainty(self, signal: Signal) -> "DFREBuilder":
        self._uncertainty = signal
        return self

    def with_extra_signal(self, name: str, signal: Signal,
                          weight: float = 1.0) -> "DFREBuilder":
        """Register an additional named signal (freshness, volume, schema, outliers...).

        At ``evaluate`` time, pass its inputs via ``context={name: args}``.
        """
        if weight < 0:
            raise ValueError("weight must be >= 0")
        self._extra[name] = signal
        self._weights[name] = float(weight)
        return self

    def with_weights(self, **weights: float) -> "DFREBuilder":
        """Per-signal weights, e.g. ``.with_weights(M=2.0, freshness=0.5)``."""
        for k, v in weights.items():
            if v < 0:
                raise ValueError("weights must be >= 0")
            self._weights[k] = float(v)
        return self

    def with_parameters(self, *, lam: float = 1.0, gamma: float = 1.0) -> "DFREBuilder":
        if lam < 0 or gamma < 0:
            raise ValueError("lam and gamma must be >= 0")
        self._lam = float(lam)
        self._gamma = float(gamma)
        return self

    def with_policy(self, policy: Optional[Policy]) -> "DFREBuilder":
        self._policy = policy
        return self

    def build(self) -> DFREPipeline:
        return DFREPipeline(
            _PipelineConfig(
                missingness=self._missingness,
                invalidity=self._invalidity,
                drift=self._drift,
                uncertainty=self._uncertainty,
                lam=self._lam,
                gamma=self._gamma,
                extra_signals=self._extra,
                weights=self._weights,
            ),
            policy=self._policy,
        )


def _row_count(data) -> int:
    if hasattr(data, "shape") and len(data.shape) > 0:
        return int(data.shape[0])
    try:
        return len(data)
    except TypeError:
        return 0
