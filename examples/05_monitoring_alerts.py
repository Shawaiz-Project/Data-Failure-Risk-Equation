"""Wiring DFRE into history, alerts, trends and reports (0.2.0)."""

import dfre
from dfre.monitoring.alerts import AlertRouter, logging_sink, file_sink
from dfre.monitoring.snapshot import make_snapshot
from dfre.monitoring.history import HistoryStore
from dfre.monitoring.trends import TrendDetector
from dfre.report.html import generate_html_report

try:
    from dfre.integrations.prometheus_ext import DFREMetrics
    HAS_PROM = True
except ImportError:
    HAS_PROM = False

router = AlertRouter()
router.add_sink(logging_sink())
router.add_sink(file_sink("alerts.jsonl"))

history = HistoryStore("dfre_history.jsonl")
metrics = DFREMetrics() if HAS_PROM else None

for batch_id, (M, V, D, U, N) in {
    "batch_001": (0.02, 0.01, 0.03, 0.05, 5000),
    "batch_002": (0.35, 0.30, 0.40, 0.55, 5000),
    "batch_003": (0.85, 0.80, 0.75, 0.90, 5000),
}.items():
    result = dfre.score(M, V, D, U, n=N)
    snapshot = make_snapshot(M, V, D, U, N, batch_id=batch_id)

    print(f"{batch_id}  R={result.risk:.4f}  {result.band}")
    history.append(result, batch_id=batch_id)
    router.dispatch(snapshot, result)
    if metrics:
        metrics.observe(batch_id, result)

# Trend analysis + HTML report
trend = TrendDetector().analyze(history.risks())
print(f"Trend: {trend.direction}  slope={trend.slope:+.4f}  alarm={trend.alarm}")

with open("dfre_report.html", "w") as fh:
    fh.write(generate_html_report(history.load(), title="Daily pipeline risk"))
print("Report written to dfre_report.html")
