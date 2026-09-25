"""Batch snapshot dataclass."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SignalSnapshot:
    batch_id: str
    timestamp: str
    M: float
    V: float
    D: float
    U: float
    N: int
    source: str = "unknown"
    model_version: str = "unknown"
    reference_version: str = "v1"
    extra_signals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def batch_id_from_array(arr) -> str:
    """Stable hash of an array or DataFrame."""
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def make_snapshot(
    M: float, V: float, D: float, U: float, N: int,
    *,
    batch_id: Optional[str] = None,
    source: str = "unknown",
    model_version: str = "unknown",
    reference_version: str = "v1",
    extra_signals: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SignalSnapshot:
    now = datetime.now(timezone.utc)
    return SignalSnapshot(
        batch_id=batch_id or f"batch_{now.timestamp():.0f}",
        timestamp=now.isoformat().replace("+00:00", "Z"),
        M=float(M), V=float(V), D=float(D), U=float(U), N=int(N),
        source=source,
        model_version=model_version,
        reference_version=reference_version,
        extra_signals=dict(extra_signals or {}),
        metadata=dict(metadata or {}),
    )
