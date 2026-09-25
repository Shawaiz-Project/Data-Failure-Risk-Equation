"""Custom exceptions for the dfre library."""

from __future__ import annotations


class DFREError(Exception):
    """Base class for all dfre errors."""


class InvalidInputError(DFREError, ValueError):
    """Raised when a DFRE input is out of range or malformed."""


class InvalidParameterError(DFREError, ValueError):
    """Raised when lambda, gamma, or another model parameter is invalid."""


class CalibrationError(DFREError):
    """Raised when calibration cannot produce usable parameters."""


class NotFittedError(DFREError, RuntimeError):
    """Raised when a fitted artifact is required but missing."""


class ConfigurationError(DFREError, ValueError):
    """Raised when a pipeline/builder configuration is inconsistent."""


class SignalComputationError(DFREError, RuntimeError):
    """Raised when a signal extractor fails on the provided data."""
