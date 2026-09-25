"""Pluggable alert routing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from ..core.types import DFREResult
from .snapshot import SignalSnapshot


AlertSink = Callable[[Dict], None]


class AlertRouter:
    """Dispatches alerts to registered sinks.

    Examples
    --------
    >>> from dfre.monitoring.alerts import AlertRouter
    >>> seen = []
    >>> router = AlertRouter()
    >>> router.add_sink(lambda payload: seen.append(payload))
    >>> router.dispatch_only
    ['MODERATE', 'HIGH', 'CRITICAL']
    """

    def __init__(
        self,
        sinks: Optional[List[AlertSink]] = None,
        dispatch_only: Optional[List[str]] = None,
    ) -> None:
        self._sinks: List[AlertSink] = list(sinks or [])
        self.dispatch_only = dispatch_only or ["MODERATE", "HIGH", "CRITICAL"]

    def add_sink(self, sink: AlertSink) -> "AlertRouter":
        self._sinks.append(sink)
        return self

    def dispatch(
        self,
        snapshot: SignalSnapshot,
        result: DFREResult,
    ) -> Optional[Dict]:
        band = result.band or "UNKNOWN"
        if band not in self.dispatch_only:
            return None

        payload = {
            "batch_id": snapshot.batch_id,
            "timestamp": snapshot.timestamp,
            "band": band,
            "action": result.action,
            "risk": result.risk,
            "components": {
                "M": snapshot.M, "V": snapshot.V,
                "D": snapshot.D, "U": snapshot.U,
                **snapshot.extra_signals,
                "pairwise": result.pairwise_component,
                "four_way": result.four_way_component,
            },
            "metadata": {**snapshot.metadata, **result.metadata},
        }

        for sink in self._sinks:
            try:
                sink(payload)
            except Exception:
                pass
        return payload


def webhook_sink(url: str, timeout: float = 5.0) -> AlertSink:
    """Create a sink that POSTs JSON to a URL."""
    import urllib.request

    def _sink(payload: Dict) -> None:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout).read()

    return _sink


def slack_sink(webhook_url: str, timeout: float = 5.0) -> AlertSink:
    """Create a sink that posts a formatted message to a Slack webhook."""
    import urllib.request

    def _sink(payload: Dict) -> None:
        text = (
            f":warning: *DFRE {payload['band']}* — batch `{payload['batch_id']}`\n"
            f"risk = {payload['risk']:.4f} → action: `{payload['action']}`"
        )
        req = urllib.request.Request(
            webhook_url,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=timeout).read()

    return _sink


def file_sink(path: Union[str, Path]) -> AlertSink:
    """Create a sink that appends each alert payload to a JSONL file."""
    target = Path(path)

    def _sink(payload: Dict) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")

    return _sink


def logging_sink(logger=None) -> AlertSink:
    """Create a sink that logs the payload."""
    import logging

    logger = logger or logging.getLogger("dfre.alerts")

    def _sink(payload: Dict) -> None:
        logger.warning("DFRE alert: %s", payload)

    return _sink
