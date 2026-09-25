# dfre — Data Failure Risk Equation

> A single-line, production-ready risk score for joint data/AI degradation — now with
> 8 signals, explainability, sensitivity analysis, escalation policies, trend detection,
> self-contained HTML reports, and a FastAPI microservice.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why

Data pipelines rarely fail for one reason. Missingness + drift + invalid records +
rising uncertainty + stale data + volume anomalies **combine** into something worse
than any single signal suggests. `dfre` quantifies that combined risk as a bounded
score **𝓡 ∈ [0, 1)**:

```
𝓡 = 1 − exp( −[ M + V + D + U + λ(MV+VD+DU+UM) + γ·MVDU ] / (1 + √N) )
```

and (v0.2.0) generalizes to **any number of named signals** with per-signal weights
and full pairwise interactions.

## Install

```bash
pip install dfre                          # zero-dependency core (numpy only)
pip install "dfre[pandas,scipy]"          # DataFrame workflows + drift statistics
pip install "dfre[serve]"                 # FastAPI microservice
pip install "dfre[all]"                   # everything
```

## Quick start

```python
import dfre

result = dfre.score(missing=0.08, invalid=0.04, drift=0.12, uncertainty=0.20)
print(result.risk, result.band, result.action)
```

## What's new in 0.2.0

| Area | Addition |
|---|---|
| Signals | `FreshnessSignal`, `VolumeSignal`, `SchemaSignal`, `OutlierSignal` |
| Equation | `dfre_generalized` — N named signals, weights, all-pairs interactions |
| XAI | `dfre.explain` — per-signal contribution shares + top driver |
| Sensitivity | `partial_derivatives`, `tornado`, `elasticity` |
| Policy | `EscalationPolicy` — consecutive-breach escalation + hysteresis |
| Monitoring | `HistoryStore` (JSONL), `TrendDetector` (EWMA/CUSUM/slope/forecast) |
| Reports | `dfre.report.generate_html_report` — self-contained HTML, zero JS deps |
| Serving | `dfre.integrations.fastapi_ext.create_app` — `/score` + `/health` |
| CLI | `score`, `explain`, `calibrate`, `calibrate-thresholds`, `trend`, `report`, `serve`, `info` |
| Calibration | optional k-fold cross-validated AUC + bootstrap CI |

See **[DOCUMENTATION.md](DOCUMENTATION.md)** for the complete guide.

## License

MIT
