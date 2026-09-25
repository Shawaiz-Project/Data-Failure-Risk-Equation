"""Append-only JSONL history store for DFRE results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..core.types import DFREResult


class HistoryStore:
    """Persist every scored batch to a JSONL file and query it later.

    Examples
    --------
    >>> import dfre, tempfile, os
    >>> path = os.path.join(tempfile.mkdtemp(), "history.jsonl")
    >>> store = HistoryStore(path)
    >>> store.append(dfre.score(0.1, 0.1, 0.1, 0.1), batch_id="b1")
    >>> store.stats()["count"]
    1
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    # ------------------------------------------------------------------
    def append(
        self,
        result: DFREResult,
        *,
        batch_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        record = {
            "batch_id": batch_id or result.metadata.get("batch_id", "unknown"),
            "timestamp": timestamp or result.metadata.get("timestamp"),
            "risk": result.risk,
            "band": result.band,
            "inputs": result.inputs,
            "extra_signals": result.extra_signals,
            "metadata": {**result.metadata, **(extra or {})},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record

    # ------------------------------------------------------------------
    def load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def risks(self) -> List[float]:
        return [float(r["risk"]) for r in self.load()]

    def stats(self, window: Optional[int] = None) -> Dict[str, Any]:
        """Rolling summary: count, mean/std/min/max of risk, band counts."""
        records = self.load()
        if window is not None:
            records = records[-window:]
        risks = [float(r["risk"]) for r in records]
        bands: Dict[str, int] = {}
        for r in records:
            band = r.get("band") or "UNKNOWN"
            bands[band] = bands.get(band, 0) + 1
        if not risks:
            return {"count": 0, "mean": None, "std": None, "min": None,
                    "max": None, "bands": bands}
        arr = np.asarray(risks)
        return {
            "count": len(risks),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "bands": bands,
        }

    def to_dataframe(self):
        """Return history as a pandas DataFrame (requires dfre[pandas])."""
        import pandas as pd
        return pd.DataFrame(self.load())
