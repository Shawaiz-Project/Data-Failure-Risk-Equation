"""Prometheus metrics exporter for DFRE."""

from __future__ import annotations

from typing import Optional

try:
    from prometheus_client import CollectorRegistry, Gauge
except ImportError as e:  # pragma: no cover
    raise ImportError("prometheus-client required; pip install dfre[prometheus]") from e

from ..core.types import DFREResult


class DFREMetrics:
    """Prometheus gauges tracking the latest DFRE values."""

    def __init__(self, registry: Optional[CollectorRegistry] = None,
                 namespace: str = "dfre") -> None:
        reg = registry
        self.risk = Gauge(f"{namespace}_risk",
                          "Latest DFRE risk score",
                          ["batch_id"], registry=reg)
        self.M = Gauge(f"{namespace}_missing",
                       "Missingness signal", ["batch_id"], registry=reg)
        self.V = Gauge(f"{namespace}_invalid",
                       "Invalidity signal", ["batch_id"], registry=reg)
        self.D = Gauge(f"{namespace}_drift",
                       "Drift signal", ["batch_id"], registry=reg)
        self.U = Gauge(f"{namespace}_uncertainty",
                       "Uncertainty signal", ["batch_id"], registry=reg)
        self.pairwise = Gauge(f"{namespace}_pairwise",
                              "Pairwise interaction contribution",
                              ["batch_id"], registry=reg)
        self.four_way = Gauge(f"{namespace}_four_way",
                              "Higher-order interaction contribution",
                              ["batch_id"], registry=reg)

    def observe(self, batch_id: str, result: Optional[DFREResult]) -> None:
        if result is None:
            return
        self.risk.labels(batch_id).set(result.risk)
        self.M.labels(batch_id).set(result.inputs.get("M", 0.0))
        self.V.labels(batch_id).set(result.inputs.get("V", 0.0))
        self.D.labels(batch_id).set(result.inputs.get("D", 0.0))
        self.U.labels(batch_id).set(result.inputs.get("U", 0.0))
        self.pairwise.labels(batch_id).set(result.pairwise_component)
        self.four_way.labels(batch_id).set(result.four_way_component)
