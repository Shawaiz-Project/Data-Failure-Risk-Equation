"""Composable builder for a realistic ETL pipeline (with the 0.2.0 extra signals)."""

import numpy as np
import pandas as pd

from dfre import DFREBuilder
from dfre.signals import (
    MissingnessSignal, InvaliditySignal, DriftSignal, UncertaintySignal,
    FreshnessSignal, VolumeSignal, SchemaSignal, OutlierSignal,
)

reference = pd.read_csv("reference.csv")   # your reference batch
incoming = pd.read_csv("incoming.csv")     # new batch
probs = np.load("predictions.npy")         # (n_samples, n_classes)

rules = {
    "age": lambda s: s.between(0, 120),
    "bmi": lambda s: s.between(10, 60),
    "label": lambda s: s.isin(["covid", "non-covid"]),
}

pipeline = (
    DFREBuilder()
    .with_missingness(MissingnessSignal())
    .with_invalidity(InvaliditySignal(rules))
    .with_drift(DriftSignal(["age", "bmi"], statistic="psi", scale=0.25))
    .with_uncertainty(UncertaintySignal(method="entropy"))
    # --- new in 0.2.0: extra signals, weighted -------------------------------
    .with_extra_signal("freshness", FreshnessSignal(sla_seconds=3600), weight=1.0)
    .with_extra_signal("volume", VolumeSignal(expected_rows=10_000), weight=0.5)
    .with_extra_signal("schema", SchemaSignal(["age", "bmi", "label"]), weight=2.0)
    .with_extra_signal("outliers", OutlierSignal(["age", "bmi"]), weight=0.5)
    .with_parameters(lam=1.0, gamma=1.0)
    .build()
)

result = pipeline.evaluate(
    incoming, reference, probs,
    context={"freshness": (incoming.attrs.get("produced_at"),)},
    metadata={"run": "daily"},
)

if result.band in {"HIGH", "CRITICAL"}:
    print(f"Blocking batch. Risk = {result.risk:.4f}  extras={result.extra_signals}")
else:
    print(f"Batch approved. Risk = {result.risk:.4f}")
