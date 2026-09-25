# Changelog

## 0.2.0 — 2026-09-25

### Added
- **New signals**: `FreshnessSignal` (data staleness vs SLA), `VolumeSignal`
  (row-count anomaly, log-ratio), `SchemaSignal` (missing/extra columns, dtype
  changes), `OutlierSignal` (IQR / z-score row flagging).
- **Generalized equation** `dfre_generalized`: arbitrary named signals in [0, 1],
  per-signal weights, full pairwise interaction sum, higher-order product term.
- **Explainability** `dfre.explain`: leave-one-out contribution shares per signal
  with top-driver identification and human-readable text.
- **Sensitivity analysis**: analytic `partial_derivatives`, `tornado` (±δ sweep),
  and `elasticity`.
- **EscalationPolicy**: consecutive-breach escalation with hysteresis to prevent
  band flapping.
- **HistoryStore**: append-only JSONL history with rolling stats and DataFrame export.
- **TrendDetector**: EWMA smoothing, linear slope, CUSUM-style alarm, naive forecast.
- **HTML reports**: `generate_html_report` — summary cards, inline SVG risk chart,
  band-colored batch table, top-driver analysis. Zero JavaScript dependencies.
- **FastAPI serving**: `create_app()` exposes `POST /score` and `GET /health`.
- **CLI**: new `explain`, `trend`, `report`, `serve` subcommands.
- **Calibration**: optional k-fold cross-validated AUC and bootstrap confidence
  intervals for the AUC of the selected (λ, γ).
- New alert sinks: `file_sink`, `slack_sink`.
- New error types: `ConfigurationError`, `SignalComputationError`.

### Changed
- `DFREResult` gains `extra_signals` and `weights` fields.
- `DFRE` facade gains `score_generalized`, optional history persistence, and
  v2 save/load format (backwards compatible with v1 files).
- `DFREBuilder` gains `with_freshness/volume/schema/outliers/weights`.
- Docstring examples corrected to exact computed values.

## 0.1.0
- Initial release: 4 signals (M, V, D, U), core equation, builder API, policy bands,
  calibration, alerts, pandas/sklearn/Airflow/Prometheus integrations, CLI.
