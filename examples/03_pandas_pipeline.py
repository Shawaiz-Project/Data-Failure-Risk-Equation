"""Using the pandas accessor for one-shot scoring (now with schema + outliers)."""

import numpy as np
import pandas as pd

import dfre.integrations.pandas_ext  # registers df.dfre

reference = pd.read_csv("reference.csv")
incoming = pd.read_csv("incoming.csv")
probs = np.load("predictions.npy")

result = incoming.dfre.score(
    reference=reference,
    rules={
        "age": lambda s: s.between(0, 120),
        "label": lambda s: s.isin(["covid", "non-covid"]),
    },
    continuous_cols=["age", "bmi"],
    drift_scale=0.25,
    statistic="psi",
    probabilities=probs,
    expected_columns=["age", "bmi", "label"],   # new: schema check
    outlier_cols=["age", "bmi"],                # new: outlier check
    lam=1.0,
    gamma=1.0,
)

print(f"DFRE = {result.risk:.4f}  band={result.band}")
print(f"extra signals: {result.extra_signals}")
