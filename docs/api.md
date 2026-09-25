# API Reference

## One-liner layer
- `dfre.score(missing, invalid, drift, uncertainty, *, n=1000, lam=1.0, gamma=1.0, policy=DEFAULT_POLICY, metadata=None) -> DFREResult`
- `dfre.score_generalized(signals: Mapping[str, float], *, weights=None, n=1000, lam=1.0, gamma=1.0, policy=DEFAULT_POLICY, metadata=None) -> DFREResult`
- `dfre.explain(signals, *, weights=None, n=1000, lam=1.0, gamma=1.0) -> Explanation`
- `dfre.score_array(M, V, D, U, *, n=1000, lam=1.0, gamma=1.0) -> np.ndarray`

## Equation layer
- `dfre.core.equation.dfre(...)` / `dfre_array(...)` — classic 4-signal form
- `dfre.core.equation.dfre_generalized(signals, *, weights, n, lam, gamma, normalize_by_n=True, return_components=False)`
- `dfre.core.equation.inverse_dfre(risk, n=1000)` — signal budget for a target risk
- `dfre.core.sensitivity.partial_derivatives / tornado / elasticity`

## Builder layer
`DFREBuilder()` → `.with_missingness/.with_invalidity/.with_drift/.with_uncertainty(signal)`
→ `.with_extra_signal(name, signal, weight=1.0)` → `.with_weights(M=2.0, ...)`
→ `.with_parameters(lam=, gamma=)` → `.with_policy(policy)` → `.build()`
→ `pipeline.evaluate(batch, reference=None, probabilities=None, context=None, metadata=None)`

## Facade
`DFRE(lam=1.0, gamma=1.0, policy=DEFAULT_POLICY, weights=None, history_path=None, metadata=None)`
with `.score(...)`, `.score_generalized(signals)`, `.explain(signals)`, `.history()`, `.save(path)`, `DFRE.load(path)`.

## Signals (all return floats in [0, 1])
| Signal | Class | Key parameters |
|---|---|---|
| Missingness M | `MissingnessSignal(sentinels=None)` | custom sentinel values |
| Invalidity V | `InvaliditySignal(rules)` | `rules={col: fn}`; `.failing_columns(df)` diagnostics |
| Drift D | `DriftSignal(cols, categorical_cols, statistic="psi", scale=0.25)` | psi / ks / wasserstein; `.raw()`, `.explain()` |
| Uncertainty U | `UncertaintySignal(method="entropy")` | entropy / margin / mc_dropout |
| Freshness F | `FreshnessSignal(sla_seconds)` | age vs SLA |
| Volume L | `VolumeSignal(expected_rows, scale=1.0)` | log-ratio anomaly |
| Schema S | `SchemaSignal(expected_columns, expected_dtypes, allow_extra)` | `.report()` diagnostics |
| Outliers O | `OutlierSignal(columns, method="iqr", threshold=None)` | iqr / zscore |

## Policy
- `Policy(low_max=0.20, moderate_max=0.45, high_max=0.70, actions=...)`
- `EscalationPolicy(base, consecutive=3, escalate_at="HIGH", cooldown=2)`

## Calibration
- `calibrate(history, lam_grid=..., gamma_grid=..., label_col="failed", n=None, cv_folds=1, bootstrap=0, seed=42)`
- `calibrate_thresholds(y_true, y_score, cost_fp=1.0, cost_fn=10.0, grid=51)` + `to_policy(result)`

## Monitoring & reporting
- `make_snapshot(M, V, D, U, N, ...)`, `SignalSnapshot`
- `AlertRouter(sinks, dispatch_only)`; sinks: `webhook_sink`, `slack_sink`, `file_sink`, `logging_sink`
- `HistoryStore(path)`: `.append`, `.load`, `.risks`, `.stats(window)`, `.to_dataframe()`
- `TrendDetector(window=20, ewma_alpha=0.3, cusum_threshold=0.5, slope_threshold=0.005)`
- `dfre.report.generate_html_report(records, title=...)`

## Integrations
`dfre.integrations.pandas_ext` (accessor `df.dfre.score`), `sklearn_ext.DFRETransformer`,
`airflow_ext.DFREScoringOperator`, `prometheus_ext.DFREMetrics`, `fastapi_ext.create_app(model)`.
