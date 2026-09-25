# DFRE — The Complete Guide

**Data Failure Risk Equation · v0.2.0 · by Shawaiz Ali · MIT License**

---

## Table of Contents

1. [The Problem DFRE Solves](#1-the-problem-dfre-solves)
2. [The Equation](#2-the-equation)
3. [Mental Model & Design Principles](#3-mental-model--design-principles)
4. [Installation](#4-installation)
5. [The Eight Signals](#5-the-eight-signals)
6. [Three API Layers](#6-three-api-layers)
7. [Explainability & Sensitivity](#7-explainability--sensitivity)
8. [Policies: Bands, Escalation, Hysteresis](#8-policies-bands-escalation-hysteresis)
9. [Calibration](#9-calibration)
10. [Monitoring: History, Trends, Reports](#10-monitoring-history-trends-reports)
11. [Alerting](#11-alerting)
12. [Serving DFRE over HTTP](#12-serving-dfre-over-http)
13. [CLI Reference](#13-cli-reference)
14. [Framework Integrations](#14-framework-integrations)
15. [Real-World Use Cases & Benefits](#15-real-world-use-cases--benefits)
16. [Best Practices & FAQ](#16-best-practices--faq)

---

## 1. The Problem DFRE Solves

Modern data and ML pipelines fail **silently and jointly**:

- A feed starts dropping rows (volume anomaly) *and* the remaining rows drift
  from the training distribution. Each issue alone might be under its alerting
  threshold — together they poison a model retrain.
- A schema change renames a column, which turns into missing values downstream,
  which inflates model uncertainty. Three "moderate" problems, one disaster.
- Data arrives 3 hours late; the batch is technically valid but statistically
  stale. Your dashboards show green while decisions are made on old data.

Traditional monitoring alerts on **one metric at a time** (great_expectations
validates, Evidently measures drift, a confidence dashboard watches uncertainty).
Nobody answers the only question the on-call engineer actually has:

> **"Is this batch safe to use — yes or no, and how close are we to trouble?"**

DFRE answers with a single bounded number **𝓡 ∈ [0, 1)** that:

| Property | Why it matters |
|---|---|
| **Bounded** | Safe for dashboards, SLOs, and automated gates — never explodes to infinity |
| **Interaction-aware** | λ and γ terms model how moderate problems *compound* each other |
| **Confidence-aware** | √(N) normalization: a signal from 50 rows is less trustworthy than from 50,000 |
| **Explainable** | Every score decomposes into per-signal contributions — you know *what* to fix |
| **Calibratable** | λ, γ and alert thresholds are learned from YOUR incident history, not hard-coded |
| **Operational** | Maps to LOW / MODERATE / HIGH / CRITICAL bands with explicit recommended actions |

### Who benefits

- **Data engineers** — gate ETL/ELT batches before they land in the warehouse.
- **ML engineers / MLOps** — guard retraining and inference against data degradation.
- **Analytics engineers** — stop broken dashboards before stakeholders see them.
- **Platform/SRE teams** — one risk metric across all pipelines, Prometheus-native.
- **Regulated industries** — an auditable, explainable, persisted risk trail.

---

## 2. The Equation

### Classic form (four core signals)

```
𝓡 = 1 − exp( −[ M + V + D + U + λ(MV+VD+DU+UM) + γ·MVDU ] / (1 + √N) )
```

| Symbol | Signal | Range | Meaning |
|---|---|---|---|
| M | Missingness | [0,1] | Fraction of expected values that are missing/sentinel |
| V | Invalidity | [0,1] | Fraction of rows failing validation rules |
| D | Drift | [0,1] | Distribution shift vs. a reference batch (PSI/KS/Wasserstein) |
| U | Uncertainty | [0,1] | Model predictive uncertainty (entropy / margin / MC-dropout) |
| N | Batch size | ≥1 | Number of observations the signals were computed from |
| λ | Pairwise strength | ≥0 | How strongly pairs of problems compound (cyclic pairs MV, VD, DU, UM) |
| γ | Higher-order strength | ≥0 | How strongly all four problems firing together compounds |

**Structure:**

- **Main term** `M+V+D+U` — each problem contributes linearly.
- **Pairwise term** `λ(MV+VD+DU+UM)` — two simultaneous moderate problems score
  *worse* than their sum. This is the core insight: 30% missing + 30% drift is
  not "0.6 of a problem", it's qualitatively more dangerous.
- **Four-way term** `γ·MVDU` — when everything degrades at once, risk spikes
  super-linearly.
- **Confidence divisor** `1+√N` — the same signals measured on a bigger batch
  are *more* statistically reliable, so a small-batch blip doesn't page anyone.
- **`1 − exp(−Z)` squashing** — bounded, monotone, interpretable as a
  "probability-like" risk that saturates gracefully.

### Generalized form (v0.2.0, any signals)

```
𝓡 = 1 − exp( −[ Σᵢ wᵢsᵢ + λ·Σᵢ<ⱼ wᵢwⱼsᵢsⱼ + γ·Πᵢ sᵢ ] / (1 + √N) )
```

Add any named signals (freshness, volume, schema, outliers, or your own),
weight them (`wᵢ`), and all pairwise interactions are included automatically.

### Inverse (capacity planning)

`inverse_dfre(risk, n)` returns the total signal budget that produces a target
risk — e.g. "how much combined signal can we tolerate before hitting HIGH?"

---

## 3. Mental Model & Design Principles

1. **Non-invasive.** DFRE adds a monitoring *layer*; it never replaces your
   validation, tests, or contracts.
2. **Composable.** Use one signal or all eight. Override any with your own
   `Signal` subclass.
3. **Calibratable.** λ, γ, thresholds come from your history — see §9.
4. **Bounded.** Output is always in [0, 1).
5. **Zero-dependency core.** `numpy` only; everything else is an optional extra.
6. **Fail-safe monitoring.** History persistence and alerting never raise into
   your pipeline — a broken monitor must never break the pipeline it watches.

---

## 4. Installation

```bash
pip install dfre                            # core (numpy only)
pip install "dfre[pandas,scipy]"            # DataFrames + KS/Wasserstein drift
pip install "dfre[calibration]"             # scikit-learn transformer + tooling
pip install "dfre[serve]"                   # FastAPI microservice
pip install "dfre[prometheus]"              # metrics exporter
pip install "dfre[airflow]"                 # Airflow operator
pip install "dfre[all]"                     # everything
```

From source: `git clone <repo> && pip install -e ".[dev]" && pytest -q`

---

## 5. The Eight Signals

All signals are classes under `dfre.signals` with a single
`compute(...) -> float in [0, 1]` method. Functional shortcuts exist too
(`missing_rate(df)`, `volume_rate(...)`, …).

### 5.1 Missingness — `MissingnessSignal(sentinels=None)`
Fraction of missing cells. Understands NaN/None plus configurable sentinels
(`-1`, `-999`, `"NA"`, `"unknown"`, …). Works on DataFrames, arrays, iterables.

```python
from dfre.signals import MissingnessSignal
M = MissingnessSignal(sentinels=["N/A", -1]).compute(batch_df)
```

### 5.2 Invalidity — `InvaliditySignal(rules)`
Fraction of rows failing any validation rule. Rules are lambdas over columns:

```python
from dfre.signals import InvaliditySignal
V = InvaliditySignal({
    "age":   lambda s: s.between(0, 120),
    "email": lambda s: s.str.contains("@"),
    "label": lambda s: s.isin(["pos", "neg"]),
}).compute(batch_df)
# diagnostics:
InvaliditySignal(rules).failing_columns(batch_df)  # per-column failure rates
```

### 5.3 Drift — `DriftSignal(continuous_cols, categorical_cols=None, statistic="psi", scale=0.25)`
Compares the batch against a reference distribution. Statistics: **PSI**
(default, no dependencies), **KS**, **Wasserstein** (scipy), plus **total
variation** and **Jensen–Shannon** for categoricals. Raw per-column values are
divided by `scale` and clipped to [0, 1].

```python
from dfre.signals import DriftSignal
sig = DriftSignal(["age", "income"], categorical_cols=["region"],
                  statistic="psi", scale=0.25)
D = sig.compute(reference_df, batch_df)
sig.explain(reference_df, batch_df)   # per-column raw + normalized values
```

Rule of thumb for `scale` with PSI: 0.1 = small shift, 0.25 = significant,
0.5+ = severe.

### 5.4 Uncertainty — `UncertaintySignal(method="entropy")`
- `"entropy"`: normalized predictive entropy of a probability matrix.
- `"margin"`: 1 − mean(top1 − top2) — cheap confusion measure.
- `"mc_dropout"`: dispersion across T stochastic forward passes.

```python
U = UncertaintySignal("entropy").compute(model.predict_proba(X_batch))
```

### 5.5 Freshness — `FreshnessSignal(sla_seconds)` *(new)*
`F = min(1, age / SLA)` — is the data too old to trust?

```python
from dfre.signals import FreshnessSignal
F = FreshnessSignal(sla_seconds=3600).compute(batch_timestamp)  # hourly SLA
```

### 5.6 Volume — `VolumeSignal(expected_rows, scale=1.0)` *(new)*
Log-ratio row-count anomaly: half or double the expected rows ≈ 0.69.

```python
from dfre.signals import VolumeSignal
L = VolumeSignal(expected_rows=50_000).compute(len(batch_df))
```

### 5.7 Schema — `SchemaSignal(expected_columns, expected_dtypes=None, allow_extra=False)` *(new)*
Fraction of structural contract checks failing (missing columns, unexpected
extras, dtype changes).

```python
from dfre.signals import SchemaSignal
S = SchemaSignal(["id", "age", "label"],
                 expected_dtypes={"age": "int64"}).compute(batch_df)
```

### 5.8 Outliers — `OutlierSignal(columns=None, method="iqr", threshold=None)` *(new)*
Fraction of rows that are outliers in ANY watched numeric column (IQR or
z-score).

```python
from dfre.signals import OutlierSignal
O = OutlierSignal(["age", "income"], method="iqr").compute(batch_df)
```

### Custom signals

```python
from dfre.signals import Signal

class DuplicateSignal(Signal):
    name = "duplicates"
    def compute(self, df):
        return float(df.duplicated().mean())
```

---

## 6. Three API Layers

### Layer 1 — One-liner

```python
import dfre

result = dfre.score(missing=0.08, invalid=0.04, drift=0.12, uncertainty=0.20, n=1000)
result.risk      # 0.0150
result.band      # 'LOW'
result.action    # 'continue_normal_monitoring'
result.pairwise_component, result.four_way_component   # decomposition
result.to_dict() # JSON-serializable
```

With extra signals (generalized):

```python
result = dfre.score_generalized(
    {"M": 0.08, "V": 0.04, "D": 0.12, "U": 0.20, "freshness": 0.5, "volume": 0.1},
    weights={"freshness": 2.0},   # staleness matters twice as much
    n=1000,
)
result.extra_signals   # {'freshness': 0.5, 'volume': 0.1}
```

Vectorized: `dfre.score_array(M_arr, V_arr, D_arr, U_arr)`.

### Layer 2 — Builder (full pipelines)

```python
from dfre import DFREBuilder
from dfre.signals import (MissingnessSignal, InvaliditySignal, DriftSignal,
                          UncertaintySignal, FreshnessSignal, SchemaSignal)

pipeline = (
    DFREBuilder()
    .with_missingness(MissingnessSignal())
    .with_invalidity(InvaliditySignal({"age": lambda s: s.between(0, 120)}))
    .with_drift(DriftSignal(["age", "income"], statistic="psi", scale=0.25))
    .with_uncertainty(UncertaintySignal("entropy"))
    .with_extra_signal("schema", SchemaSignal(["age", "income", "label"]), weight=2.0)
    .with_extra_signal("freshness", FreshnessSignal(sla_seconds=3600))
    .with_parameters(lam=1.0, gamma=1.0)
    .build()
)

result = pipeline.evaluate(
    batch_df, reference_df, probs,
    context={"freshness": (produced_at,)},   # extra-signal inputs by name
    metadata={"run_id": "2026-09-25"},
)
```

Missing optional inputs degrade gracefully: no reference → D=0; no
probabilities → U=0.

### Layer 3 — Stateful facade (production)

```python
from dfre import DFRE

model = DFRE(lam=1.0, gamma=1.0, history_path="dfre_history.jsonl")
result = model.score(0.08, 0.04, 0.12, 0.20)     # auto-appended to history
result = model.score_generalized({"M": 0.1, "V": 0.2, "freshness": 0.9})
print(model.explain({"M": 0.1, "V": 0.2, "freshness": 0.9}).text)
print(model.history().stats(window=30))

model.save("dfre_model.json")                    # freeze parameters + policy
model = DFRE.load("dfre_model.json")             # load in production
```

---

## 7. Explainability & Sensitivity

A risk score nobody can explain gets ignored. DFRE ships three lenses:

```python
import dfre
from dfre.core.sensitivity import partial_derivatives, tornado, elasticity

signals = {"M": 0.30, "V": 0.10, "D": 0.45, "U": 0.20, "freshness": 0.60}
```

**Attribution — "why is risk high?"**
```python
e = dfre.explain(signals)
e.top_driver        # 'freshness'
e.shares            # {'freshness': 0.41, 'D': 0.28, ...} — sums to 1
e.text              # "Risk 𝓡 = 0.0189 is driven mainly by freshness (41%)..."
```
Uses leave-one-out: zero each signal, measure the risk drop. Interaction terms
are credited to the signals involved in them.

**Gradients — "what should we fix first?"**
```python
grads = partial_derivatives(signals)   # exact analytic ∂𝓡/∂sᵢ
```
The largest gradient is the signal where improvement reduces risk fastest —
including its interaction effects with everything else currently wrong.

**Tornado & elasticity — "how sensitive are we?"**
```python
tornado(signals, delta=0.1)   # risk swing per signal, sorted by impact
elasticity(signals)           # %Δrisk per 1%Δsignal
```

---

## 8. Policies: Bands, Escalation, Hysteresis

### Static bands

```python
from dfre import Policy

policy = Policy(low_max=0.20, moderate_max=0.45, high_max=0.70,
                actions={"HIGH": "quarantine_and_slack_data_team"})
policy.classify(0.62)     # 'HIGH'
policy.action_for("HIGH") # 'quarantine_and_slack_data_team'
```

Defaults: `LOW → continue_normal_monitoring`, `MODERATE → open_investigation_alert`,
`HIGH → human_review_and_quarantine`, `CRITICAL → block_batch_and_page_oncall`.

### Escalation policy (anti-flapping, stateful)

Raw per-batch bands flap when risk hovers near a threshold. `EscalationPolicy`
escalates after **N consecutive** breaches and holds the elevated band until
**M consecutive** quiet batches (hysteresis):

```python
from dfre import EscalationPolicy

ep = EscalationPolicy(policy, consecutive=3, escalate_at="HIGH", cooldown=2)
# 3 HIGH batches in a row -> effective band becomes CRITICAL
# then stays CRITICAL until 2 batches in a row fall below HIGH
ep.reset()  # after the incident is resolved
```

---

## 9. Calibration

Never trust default parameters in a new domain — calibrate from incidents.

```python
from dfre import calibrate, calibrate_thresholds
from dfre.calibration.thresholds import to_policy
from dfre.core.equation import dfre_array

# history: one row per past batch, columns M, V, D, U, failed (0/1)
params = calibrate(history, label_col="failed",
                   cv_folds=5,        # cross-validated (λ, γ) selection
                   bootstrap=500)     # 95% CI for the winning AUC
print(params.lam, params.gamma, params.auc, params.auc_ci)

scores = dfre_array(history.M, history.V, history.D, history.U,
                    lam=params.lam, gamma=params.gamma)
thr = calibrate_thresholds(history.failed, scores,
                           cost_fp=1.0,    # cost of a false alarm
                           cost_fn=10.0)   # cost of a missed failure
policy = to_policy(thr)

from dfre import DFRE
DFRE(lam=params.lam, gamma=params.gamma, policy=policy).save("dfre_model.json")
```

- `cost_fn / cost_fp` encodes your business asymmetry (missed fraud ≈ 20–100×
  a false alarm).
- If interactions don't help, calibration legitimately selects λ=γ=0.
- Recalibrate monthly, or when `TrendDetector` flags a regime change.

Full guide: [`docs/calibration.md`](docs/calibration.md).

---

## 10. Monitoring: History, Trends, Reports

```python
from dfre import HistoryStore, TrendDetector, generate_html_report

store = HistoryStore("dfre_history.jsonl")          # append-only JSONL
store.append(result, batch_id="batch_2026_09_25")

store.stats(window=30)      # count, mean/std/min/max, band distribution
store.to_dataframe()        # pandas export

trend = TrendDetector(window=20).analyze(store.risks())
trend.direction   # 'increasing' | 'decreasing' | 'stable'
trend.slope       # risk units per batch
trend.ewma        # smoothed level
trend.cusum       # cumulative upward deviation
trend.alarm       # True -> investigate BEFORE thresholds breach
trend.forecast    # naive linear forecast, next 3 batches

html = generate_html_report(store.load(), title="Nightly ETL risk")
open("report.html", "w").write(html)   # self-contained: inline SVG, no JS
```

The report contains summary cards, a band-colored risk chart, top-driver
analysis per batch, and a full batch table — shareable as a single file.

---

## 11. Alerting

```python
from dfre import AlertRouter, logging_sink, webhook_sink, slack_sink, file_sink
from dfre import make_snapshot

router = AlertRouter(dispatch_only=["MODERATE", "HIGH", "CRITICAL"])
router.add_sink(logging_sink())
router.add_sink(slack_sink("https://hooks.slack.com/services/..."))
router.add_sink(webhook_sink("https://alerts.internal/dfre"))
router.add_sink(file_sink("alerts.jsonl"))

router.dispatch(make_snapshot(M, V, D, U, N, batch_id="b42"), result)
```

- Sinks are plain callables — wrap PagerDuty, Opsgenie, email, anything.
- Sink exceptions are swallowed: **alerting can never crash your pipeline**.
- Payload includes batch id, band, action, risk, all signal components, metadata.

---

## 12. Serving DFRE over HTTP

```bash
pip install "dfre[serve]"
dfre serve --port 8000 --model dfre_model.json
```

```
POST /score   {"signals": {"M": 0.1, "V": 0.2, "D": 0.1, "U": 0.3}, "n": 5000}
GET  /health  -> {"status": "ok", "lam": 1.0, "gamma": 1.0}
```

Or embed in your own app:

```python
from dfre import DFRE
from dfre.integrations.fastapi_ext import create_app
import uvicorn

uvicorn.run(create_app(DFRE.load("dfre_model.json")), port=8000)
```

---

## 13. CLI Reference

```bash
dfre score --missing 0.08 --invalid 0.04 --drift 0.12 --uncertainty 0.20 [--json]
dfre explain --signal M=0.4 --signal V=0.1 --signal freshness=0.6 [--json]
dfre calibrate --history history.csv --label failed --cv-folds 5 --bootstrap 500
dfre calibrate-thresholds --history history.csv --cost-fp 1 --cost-fn 10
dfre trend --history dfre_history.jsonl --window 20     # exit code 2 on alarm
dfre report --history dfre_history.jsonl --out report.html --title "Nightly ETL"
dfre serve --port 8000 --model dfre_model.json
dfre info
```

`dfre trend` returning exit code 2 makes it a drop-in CI/cron gate.

---

## 14. Framework Integrations

| Framework | Usage | Extra |
|---|---|---|
| **pandas** | `import dfre.integrations.pandas_ext` then `df.dfre.score(reference=..., rules=..., probabilities=..., expected_columns=..., outlier_cols=...)` | `pandas` |
| **scikit-learn** | `DFRETransformer(lam, gamma)` appends a `dfre_risk` feature inside any `Pipeline` | `calibration` |
| **Airflow** | `DFREScoringOperator(batch_loader=..., signals_fn=..., alert_router=...)` pushes results to XCom | `airflow` |
| **Prometheus** | `DFREMetrics().observe(batch_id, result)` exposes `dfre_risk`, per-signal and interaction gauges | `prometheus` |
| **FastAPI** | `create_app(model)` — see §12 | `serve` |

---

## 15. Real-World Use Cases & Benefits

### ETL/ELT gate (block bad loads)
```python
result = pipeline.evaluate(incoming_df, reference_df)
if result.band in {"HIGH", "CRITICAL"}:
    quarantine(incoming_df); raise AirflowFailException(result.action)
load_to_warehouse(incoming_df)
```
**Benefit:** broken data never reaches dashboards or models; quarantine instead
of silent corruption.

### ML retraining guard
Only retrain when today's data is trustworthy *and* similar enough to training
data. DFRE's D and U signals plus interaction terms catch the "data drifted AND
the model is unsure" double failure that pure drift monitors miss.

### Inference-side monitoring
Score each scoring batch with probabilities → uncertainty, freshness, volume.
**Benefit:** one number drives the PagerDuty rotation instead of five dashboards.

### SLA/SLO reporting
`HistoryStore` + `generate_html_report` → a weekly one-file report per pipeline:
mean/max risk, band distribution, top drivers. **Benefit:** data quality becomes
a measured, reviewable service level.

### Multi-tenant platform metric
Because 𝓡 is bounded and dimensionless, a platform team can rank hundreds of
pipelines on one axis and allocate attention where the risk actually is.

### Regulatory / audit trail
Every score is persisted (JSONL) with inputs, decomposition, band, and metadata.
**Benefit:** a defensible, explainable record of data-risk decisions.

---

## 16. Best Practices & FAQ

**What values should λ and γ have?** Start at 1.0/1.0, then calibrate (§9). If
your failures are mostly single-cause, calibration will pull them toward 0.

**What is a good `scale` for DriftSignal?** With PSI: 0.25 is a sound default
(significant shift = full signal). Lower it for sensitive domains.

**Why did my small batch score LOW even though 20% of values are missing?**
That's the √(N) confidence divisor working: with N=25 the evidence is weak. If
small batches still matter, pass a larger effective `n` or set
`normalize_by_n=False` in `dfre_generalized`.

**Does DFRE replace Great Expectations / dbt tests?** No — it *consumes* their
outcomes (as V) and combines them with drift, uncertainty, freshness, etc.

**How do I add my own signal?** Subclass `Signal`, return [0, 1], register with
`.with_extra_signal(name, signal, weight=...)`.

**Alert fatigue?** Use `EscalationPolicy` (hysteresis), restrict
`dispatch_only`, and let `TrendDetector` handle slow-burn degradation that never
crosses a hard threshold.

**Performance?** The equation is O(k²) in the number of signals and vectorized
over batches; scoring a batch is microseconds. Signal extraction dominates and
is linear in rows.

---

*Generated documentation for dfre v0.2.0. Source of truth: the docstrings in
`src/dfre/` and the runnable scripts in `examples/`.*
