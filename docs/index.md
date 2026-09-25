# dfre — Data Failure Risk Equation

**One bounded number that tells you whether your data batch is safe to use.**

Data pipelines rarely fail for a single reason. Missing values, schema changes,
invalid records, distribution drift, volume anomalies, stale data and rising model
uncertainty interact — two moderate problems together are usually worse than one
severe problem alone. DFRE models exactly that:

```
R = 1 − exp( −[ M + V + D + U + λ(MV+VD+DU+UM) + γ·MVDU ] / (1 + √N) )
```

- Bounded in [0, 1) — safe for dashboards and SLOs
- Interaction terms capture compounding failure modes
- √(N) normalization encodes statistical confidence
- Calibratable to your own incident history
- Explainable: per-signal attribution, gradients, tornado analysis

## Contents

1. [Quickstart](quickstart.md)
2. [API reference](api.md)
3. [Calibration guide](calibration.md)
