"""Operational bands and actions."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


DEFAULT_ACTIONS: Dict[str, str] = {
    "LOW": "continue_normal_monitoring",
    "MODERATE": "open_investigation_alert",
    "HIGH": "human_review_and_quarantine",
    "CRITICAL": "block_batch_and_page_oncall",
}


@dataclass
class Policy:
    """Threshold bands for converting 𝓡 into an operational band.

    Parameters
    ----------
    low_max, moderate_max, high_max : float
        Upper bounds for LOW, MODERATE, HIGH. Anything above high_max is
        classified as CRITICAL.
    actions : dict, optional
        Override default recommended actions per band.
    """

    low_max: float = 0.20
    moderate_max: float = 0.45
    high_max: float = 0.70
    actions: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ACTIONS))

    def __post_init__(self) -> None:
        if not (0 <= self.low_max <= self.moderate_max <= self.high_max <= 1):
            raise ValueError("thresholds must satisfy 0 ≤ low ≤ moderate ≤ high ≤ 1")

    def classify(self, risk: float) -> str:
        if risk < self.low_max:
            return "LOW"
        if risk < self.moderate_max:
            return "MODERATE"
        if risk < self.high_max:
            return "HIGH"
        return "CRITICAL"

    def action_for(self, band: str) -> str:
        return self.actions.get(band, "unknown")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        return cls(
            low_max=data.get("low_max", 0.20),
            moderate_max=data.get("moderate_max", 0.45),
            high_max=data.get("high_max", 0.70),
            actions=data.get("actions") or dict(DEFAULT_ACTIONS),
        )


DEFAULT_POLICY = Policy()
