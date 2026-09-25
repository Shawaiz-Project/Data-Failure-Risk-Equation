"""Schema signal (S) — structural contract violations of a batch."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .base import Signal


class SchemaSignal(Signal):
    """Fraction of schema checks that fail.

    Checks (each weighted equally):
      - every expected column is present,
      - no unexpected extra columns (when ``allow_extra=False``),
      - column dtypes match expected dtypes (when ``expected_dtypes`` given,
        uses pandas ``dtype`` names).

    Parameters
    ----------
    expected_columns : iterable of str
    expected_dtypes : dict of str → str, optional
        e.g. ``{"age": "int64", "label": "object"}``.
    allow_extra : bool, default False
        Whether columns beyond ``expected_columns`` are tolerated.
    """

    name = "schema"

    def __init__(
        self,
        expected_columns: Iterable[str],
        expected_dtypes: Optional[Dict[str, str]] = None,
        allow_extra: bool = False,
    ) -> None:
        self.expected_columns = list(expected_columns)
        self.expected_dtypes = dict(expected_dtypes or {})
        self.allow_extra = bool(allow_extra)

    def compute(self, data) -> float:
        if not self.expected_columns:
            return 0.0
        report = self.report(data)
        checks = report["checks"]
        if not checks:
            return 0.0
        failed = sum(1 for ok in checks.values() if not ok)
        return float(failed / len(checks))

    def report(self, data) -> Dict[str, object]:
        """Diagnostic: which specific checks failed."""
        columns = list(getattr(data, "columns", []))
        checks: Dict[str, bool] = {}

        for col in self.expected_columns:
            checks[f"present:{col}"] = col in columns
        if not self.allow_extra:
            extras = [c for c in columns if c not in self.expected_columns]
            checks["no_extra_columns"] = not extras
        if self.expected_dtypes and hasattr(data, "dtypes"):
            dtypes = {str(k): str(v) for k, v in data.dtypes.items()}
            for col, expected in self.expected_dtypes.items():
                checks[f"dtype:{col}"] = dtypes.get(col) == expected
        return {"checks": checks}


def schema_report(data, expected_columns: Iterable[str]) -> Dict[str, object]:
    """Functional shortcut returning the detailed check report."""
    return SchemaSignal(expected_columns).report(data)
