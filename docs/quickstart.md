# Quickstart

## Install

```bash
pip install "dfre[pandas,scipy]"
```

## 60 seconds

```python
import dfre

result = dfre.score(missing=0.08, invalid=0.04, drift=0.12, uncertainty=0.20)
print(result.risk, result.band, result.action)
```

## 5 minutes — real data

```python
import dfre
from dfre.signals import MissingnessSignal, DriftSignal

M = MissingnessSignal().compute(batch_df)
D = DriftSignal(["age", "income"], statistic="psi").compute(reference_df, batch_df)

result = dfre.score(M, V=0.0, drift=D, uncertainty=0.0, n=len(batch_df))
if result.band in {"HIGH", "CRITICAL"}:
    raise SystemExit(f"Batch blocked: {result.action}")
```

## 15 minutes — full pipeline with extras + explanation

```python
from dfre import DFREBuilder
from dfre.signals import (
    MissingnessSignal, InvaliditySignal, DriftSignal, UncertaintySignal,
    FreshnessSignal, SchemaSignal,
)

pipeline = (
    DFREBuilder()
    .with_invalidity(InvaliditySignal({"age": lambda s: s.between(0, 120)}))
    .with_drift(DriftSignal(["age"], scale=0.25))
    .with_extra_signal("schema", SchemaSignal(["age", "label"]), weight=2.0)
    .build()
)

result = pipeline.evaluate(batch_df, reference_df, probs)
print(result.risk, result.band)
print(result.extra_signals)

# Why?
print(dfre.explain(result.inputs | result.extra_signals).text)
```

## Next steps

- [Calibration](calibration.md) — fit λ, γ and thresholds to your history
- [API reference](api.md) — every public symbol
- CLI: `dfre score`, `dfre explain`, `dfre calibrate`, `dfre trend`, `dfre report`, `dfre serve`
