"""Signal extractors: each maps a raw data property to a normalized [0, 1] score."""

from .base import Signal
from .missingness import MissingnessSignal, missing_rate
from .invalidity import InvaliditySignal, invalid_rate
from .drift import DriftSignal, psi, ks, wasserstein, total_variation, jensen_shannon
from .uncertainty import UncertaintySignal, normalized_entropy, mc_dropout_uncertainty
from .freshness import FreshnessSignal, freshness_rate
from .volume import VolumeSignal, volume_rate
from .schema import SchemaSignal, schema_report
from .outliers import OutlierSignal, outlier_rate

__all__ = [
    "Signal",
    "MissingnessSignal", "missing_rate",
    "InvaliditySignal", "invalid_rate",
    "DriftSignal", "psi", "ks", "wasserstein", "total_variation", "jensen_shannon",
    "UncertaintySignal", "normalized_entropy", "mc_dropout_uncertainty",
    "FreshnessSignal", "freshness_rate",
    "VolumeSignal", "volume_rate",
    "SchemaSignal", "schema_report",
    "OutlierSignal", "outlier_rate",
]
