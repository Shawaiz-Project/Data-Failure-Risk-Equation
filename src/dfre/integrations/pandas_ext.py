"""Pandas accessor: `df.dfre.score(...)`."""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("pandas required; pip install dfre[pandas]") from e

from ..core.types import DFREResult
from ..signals.drift import DriftSignal
from ..signals.invalidity import InvaliditySignal
from ..signals.missingness import MissingnessSignal
from ..signals.uncertainty import UncertaintySignal
from ..signals.outliers import OutlierSignal
from ..signals.schema import SchemaSignal
from ..api.one_liner import score as _score, score_generalized as _score_gen


@pd.api.extensions.register_dataframe_accessor("dfre")
class DFREAccessor:
    """DataFrame accessor for one-shot DFRE scoring.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [1, None, 3]})
    >>> df.dfre.score().risk >= 0
    True
    """

    def __init__(self, pandas_obj: "pd.DataFrame") -> None:
        self._obj = pandas_obj

    def score(
        self,
        *,
        reference: Optional["pd.DataFrame"] = None,
        rules: Optional[Dict[str, Any]] = None,
        continuous_cols: Optional[list] = None,
        categorical_cols: Optional[list] = None,
        drift_scale: float = 0.25,
        statistic: str = "psi",
        probabilities=None,
        expected_columns: Optional[list] = None,
        outlier_cols: Optional[list] = None,
        outlier_method: str = "iqr",
        n: Optional[int] = None,
        lam: float = 1.0,
        gamma: float = 1.0,
        policy=None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DFREResult:
        M = MissingnessSignal().compute(self._obj)
        V = InvaliditySignal(rules or {}).compute(self._obj)

        D = 0.0
        if reference is not None and (continuous_cols or categorical_cols):
            D = DriftSignal(
                continuous_cols or [],
                categorical_cols or [],
                statistic=statistic,
                scale=drift_scale,
            ).compute(reference, self._obj)

        U = 0.0
        if probabilities is not None:
            U = UncertaintySignal().compute(probabilities)

        extras: Dict[str, float] = {}
        if expected_columns:
            extras["schema"] = SchemaSignal(expected_columns).compute(self._obj)
        if outlier_cols:
            extras["outliers"] = OutlierSignal(outlier_cols, method=outlier_method) \
                .compute(self._obj)

        n_rows = n or len(self._obj)
        if extras:
            return _score_gen(
                {"M": M, "V": V, "D": D, "U": U, **extras},
                n=n_rows, lam=lam, gamma=gamma, policy=policy, metadata=metadata,
            )
        return _score(
            M, V, D, U, n=n_rows, lam=lam, gamma=gamma,
            policy=policy, metadata=metadata,
        )
