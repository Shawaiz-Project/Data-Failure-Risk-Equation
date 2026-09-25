# Calibration Guide

DFRE has two families of tunables:

1. **Equation parameters** `λ` (pairwise interaction strength) and `γ`
   (higher-order interaction strength).
2. **Policy thresholds** `low_max / moderate_max / high_max`.

Neither is a universal constant — calibrate both against your own history of
batches that did or did not cause downstream failures.

## Step 1 — Collect history

Build a table with one row per historical batch:

| M | V | D | U | failed |
|---|---|---|---|--------|
| 0.02 | 0.01 | 0.03 | 0.05 | 0 |
| 0.35 | 0.30 | 0.40 | 0.55 | 1 |

`failed = 1` means the batch actually caused a problem (bad model retrain,
broken dashboard, downstream job failure, manual rollback...).

## Step 2 — Fit λ and γ

```python
from dfre import calibrate

params = calibrate(
    history,
    lam_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    gamma_grid=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    label_col="failed",
    cv_folds=5,      # cross-validated selection (recommended)
    bootstrap=500,   # 95% CI for the winning AUC
)
print(params.lam, params.gamma, params.auc, params.auc_ci)
```

Guidance:
- If interactions don't improve AUC, calibration will pick `λ = γ = 0`
  (pure additive model). That is a legitimate outcome.
- `cv_folds=5` guards against overfitting small histories.
- With < 30 rows, prefer wide grids and `bootstrap` to gauge uncertainty.

## Step 3 — Fit band thresholds by cost

```python
from dfre import calibrate_thresholds
from dfre.calibration.thresholds import to_policy
from dfre.core.equation import dfre_array

scores = dfre_array(history.M, history.V, history.D, history.U,
                    lam=params.lam, gamma=params.gamma)
result = calibrate_thresholds(history.failed, scores,
                              cost_fp=1.0,   # cost of a false alarm
                              cost_fn=10.0)  # cost of a missed failure
policy = to_policy(result)
```

Set `cost_fn / cost_fp` to your real asymmetry: in fraud or healthcare a missed
failure is often 20–100× more expensive than a false alarm.

## Step 4 — Freeze and ship

```python
from dfre import DFRE
model = DFRE(lam=params.lam, gamma=params.gamma, policy=policy)
model.save("dfre_model.json")
# production:
model = DFRE.load("dfre_model.json")
```

## Step 5 — Recalibrate periodically

Data drift means yesterday's calibration decays. Re-run calibration monthly or
whenever `TrendDetector` reports a sustained direction change.
