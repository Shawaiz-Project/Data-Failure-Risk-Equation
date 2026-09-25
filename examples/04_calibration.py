"""Calibrate lambda, gamma, and thresholds from historical batches.

New in 0.2.0: cross-validated model selection and bootstrap confidence intervals.
"""

import pandas as pd

from dfre import calibrate, calibrate_thresholds
from dfre.core.equation import dfre_array

history = pd.read_csv("batch_history.csv")
# columns: M, V, D, U, N, failed

params = calibrate(
    history, label_col="failed",
    cv_folds=5,        # select by mean 5-fold AUC
    bootstrap=500,     # 95% CI for the winning AUC
)
print(f"Best lambda = {params.lam}  gamma = {params.gamma}")
print(f"AUC = {params.auc:.4f}  95% CI = {params.auc_ci}")

scores = dfre_array(
    history["M"].to_numpy(), history["V"].to_numpy(),
    history["D"].to_numpy(), history["U"].to_numpy(),
    n=1000, lam=params.lam, gamma=params.gamma,
)

thresholds = calibrate_thresholds(
    history["failed"].to_numpy(), scores, cost_fp=1.0, cost_fn=10.0,
)
print(f"LOW      < {thresholds.low_max:.3f}")
print(f"MODERATE < {thresholds.moderate_max:.3f}")
print(f"HIGH     < {thresholds.high_max:.3f}")
print(f"CRITICAL >= {thresholds.high_max:.3f}")
