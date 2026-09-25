"""Apache Airflow operator for batch DFRE evaluation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

try:
    from airflow.models import BaseOperator
except ImportError as e:  # pragma: no cover
    raise ImportError("apache-airflow required; pip install dfre[airflow]") from e

from ..api.facade import DFRE
from ..monitoring.alerts import AlertRouter
from ..monitoring.snapshot import make_snapshot


class DFREScoringOperator(BaseOperator):
    """Airflow operator that scores a batch and pushes the result to XCom.

    Parameters
    ----------
    batch_loader : callable
        Returns the batch to score.
    reference_loader : callable, optional
        Returns the reference DataFrame.
    signals_fn : callable
        Signature ``(batch, reference) -> dict(M, V, D, U)``.
    dfre : DFRE, optional
        A pre-configured DFRE facade. If None, uses defaults.
    alert_router : AlertRouter, optional
    """

    template_fields = ("batch_id",)

    def __init__(
        self,
        *,
        batch_loader: Callable[[], Any],
        signals_fn: Callable[[Any, Optional[Any]], Dict[str, float]],
        reference_loader: Optional[Callable[[], Any]] = None,
        dfre: Optional[DFRE] = None,
        alert_router: Optional[AlertRouter] = None,
        batch_id: str = "batch",
        source: str = "airflow",
        model_version: str = "unknown",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.batch_loader = batch_loader
        self.reference_loader = reference_loader
        self.signals_fn = signals_fn
        self.dfre = dfre or DFRE()
        self.alert_router = alert_router or AlertRouter()
        self.batch_id = batch_id
        self.source = source
        self.model_version = model_version

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        batch = self.batch_loader()
        reference = self.reference_loader() if self.reference_loader else None

        signals = self.signals_fn(batch, reference)
        result = self.dfre.score(
            signals["M"], signals["V"], signals["D"], signals["U"],
            n=signals.get("N", len(batch)),
        )

        snapshot = make_snapshot(
            signals["M"], signals["V"], signals["D"], signals["U"],
            N=signals.get("N", len(batch)),
            batch_id=self.batch_id,
            source=self.source,
            model_version=self.model_version,
        )
        self.alert_router.dispatch(snapshot, result)

        payload = {**snapshot.to_dict(), **result.to_dict()}
        context["ti"].xcom_push(key="dfre_result", value=payload)
        return payload
