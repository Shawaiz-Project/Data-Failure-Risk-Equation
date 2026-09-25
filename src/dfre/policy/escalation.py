"""Escalation policy — band assignment with memory.

Plain :class:`~dfre.policy.bands.Policy` classifies each batch independently,
which makes alerts flap when risk hovers near a threshold. ``EscalationPolicy``
adds two behaviours:

- **Escalation**: if the raw band stays at or above ``escalate_at`` for
  ``consecutive`` batches in a row, the effective band is raised one level.
- **Hysteresis**: once elevated, the effective band only drops after the raw
  band has stayed strictly below the elevated band's lower threshold for
  ``cooldown`` consecutive batches.
"""

from __future__ import annotations

from typing import List, Optional

from .bands import Policy

_BAND_ORDER: List[str] = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


class EscalationPolicy:
    """Stateful wrapper around :class:`Policy` with escalation + hysteresis.

    Examples
    --------
    >>> from dfre.policy.escalation import EscalationPolicy
    >>> ep = EscalationPolicy(consecutive=2, escalate_at="HIGH")
    >>> ep.classify(0.8)  # first HIGH — not yet escalated
    'HIGH'
    >>> ep.classify(0.8)  # second consecutive HIGH → escalated
    'CRITICAL'
    """

    def __init__(
        self,
        base: Optional[Policy] = None,
        *,
        consecutive: int = 3,
        escalate_at: str = "HIGH",
        cooldown: int = 2,
    ) -> None:
        if consecutive < 1:
            raise ValueError("consecutive must be >= 1")
        if escalate_at not in _BAND_ORDER:
            raise ValueError(f"escalate_at must be one of {_BAND_ORDER}")
        self.base = base or Policy()
        self.consecutive = int(consecutive)
        self.escalate_at = escalate_at
        self.cooldown = max(1, int(cooldown))
        self._streak = 0
        self._elevated: Optional[str] = None
        self._cool = 0

    # ------------------------------------------------------------------
    def classify(self, risk: float) -> str:
        raw = self.base.classify(risk)
        raw_rank = _BAND_ORDER.index(raw)

        # Track consecutive breaches of the escalation floor.
        if raw_rank >= _BAND_ORDER.index(self.escalate_at):
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= self.consecutive:
            self._elevated = _BAND_ORDER[min(raw_rank + 1, len(_BAND_ORDER) - 1)]
            self._cool = 0
            return self._elevated  # triggering batch is never a cooldown batch

        # Hysteresis: hold the elevated band until `cooldown` quiet batches.
        if self._elevated is not None:
            floor_rank = _BAND_ORDER.index(self._elevated)
            if raw_rank < floor_rank:  # hysteresis countdown on quiet batches
                self._cool += 1
                if self._cool >= self.cooldown:
                    self._elevated = None
                    self._cool = 0
                    return raw
            else:
                self._cool = 0
            return self._elevated

        return raw

    def action_for(self, band: str) -> str:
        return self.base.action_for(band)

    def reset(self) -> None:
        """Forget all escalation state (e.g. after an incident is resolved)."""
        self._streak = 0
        self._elevated = None
        self._cool = 0
