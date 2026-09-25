"""dfre — Data Failure Risk Equation.

Quick start
-----------
>>> import dfre
>>> result = dfre.score(0.08, 0.04, 0.12, 0.20)
>>> result.band
'LOW'
>>> e = dfre.explain({"M": 0.4, "V": 0.1, "D": 0.1, "U": 0.1})
>>> e.top_driver
'M'
"""

from __future__ import annotations

from ._version import __version__
from .api.facade import DFRE
from .api.one_liner import score, score_array, score_generalized, explain
from .api.builder import DFREBuilder, DFREPipeline
from .core.equation import dfre, dfre_array, dfre_generalized, inverse_dfre
from .core.sensitivity import partial_derivatives, tornado, elasticity
from .core.types import DFREResult, SignalSet, Explanation
from .core.errors import (
    DFREError,
    InvalidInputError,
    InvalidParameterError,
    CalibrationError,
    NotFittedError,
    ConfigurationError,
    SignalComputationError,
)
from .policy.bands import Policy, DEFAULT_POLICY
from .policy.escalation import EscalationPolicy
from .signals.missingness import MissingnessSignal
from .signals.invalidity import InvaliditySignal
from .signals.drift import DriftSignal
from .signals.uncertainty import UncertaintySignal
from .signals.freshness import FreshnessSignal
from .signals.volume import VolumeSignal
from .signals.schema import SchemaSignal
from .signals.outliers import OutlierSignal
from .calibration.parameters import calibrate, CalibrationResult
from .calibration.thresholds import calibrate_thresholds, ThresholdResult
from .monitoring.snapshot import SignalSnapshot, make_snapshot
from .monitoring.alerts import AlertRouter, webhook_sink, slack_sink, file_sink, logging_sink
from .monitoring.history import HistoryStore
from .monitoring.trends import TrendDetector, TrendReport
from .report.html import generate_html_report


__all__ = [
    "__version__",
    # Facades & one-liners
    "DFRE", "score", "score_array", "score_generalized", "explain",
    "dfre", "dfre_array", "dfre_generalized", "inverse_dfre",
    # Sensitivity
    "partial_derivatives", "tornado", "elasticity",
    # Builder
    "DFREBuilder", "DFREPipeline",
    # Types
    "DFREResult", "SignalSet", "Explanation",
    # Errors
    "DFREError", "InvalidInputError", "InvalidParameterError",
    "CalibrationError", "NotFittedError", "ConfigurationError",
    "SignalComputationError",
    # Policy
    "Policy", "DEFAULT_POLICY", "EscalationPolicy",
    # Signals
    "MissingnessSignal", "InvaliditySignal", "DriftSignal", "UncertaintySignal",
    "FreshnessSignal", "VolumeSignal", "SchemaSignal", "OutlierSignal",
    # Calibration
    "calibrate", "CalibrationResult", "calibrate_thresholds", "ThresholdResult",
    # Monitoring
    "SignalSnapshot", "make_snapshot",
    "AlertRouter", "webhook_sink", "slack_sink", "file_sink", "logging_sink",
    "HistoryStore", "TrendDetector", "TrendReport",
    # Reporting
    "generate_html_report",
]
