"""End-to-end DFRE experiment using generated reference and incoming batches.

Install with ``python -m pip install -e ".[pandas,calibration]"`` and run
with ``python experiment.py`` from the repository root.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import dfre
import dfre.integrations.pandas_ext  # Registers the ``DataFrame.dfre`` accessor.
from dfre.calibration.thresholds import to_policy
from dfre.integrations.sklearn_ext import DFRETransformer


OUTPUT_DIR = Path(__file__).with_name("experiment_output")


def make_batches(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    reference_size = 500
    incoming_size = 420

    reference = pd.DataFrame({
        "age": rng.normal(42, 12, reference_size).clip(18, 85),
        "bmi": rng.normal(27, 5, reference_size).clip(15, 45),
        "plan": rng.choice(["basic", "plus", "premium"], reference_size,
                           p=[0.55, 0.30, 0.15]),
    })
    incoming = pd.DataFrame({
        "age": rng.normal(48, 15, incoming_size).clip(18, 95),
        "bmi": rng.normal(30, 7, incoming_size).clip(12, 60),
        "plan": rng.choice(["basic", "plus", "premium"], incoming_size,
                           p=[0.35, 0.40, 0.25]),
    })

    incoming.loc[rng.choice(incoming_size, 18, replace=False), "age"] = np.nan
    incoming.loc[rng.choice(incoming_size, 8, replace=False), "bmi"] = 92.0
    incoming.loc[rng.choice(incoming_size, 4, replace=False), "age"] = -4.0

    confidence = rng.uniform(0.55, 0.98, incoming_size)
    probabilities = np.column_stack([confidence, 1.0 - confidence])
    return reference, incoming, probabilities


def make_calibration_history(seed: int = 19, rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    history = pd.DataFrame(rng.uniform(0.0, 0.8, size=(rows, 4)),
                           columns=["M", "V", "D", "U"])
    signals = history[["M", "V", "D", "U"]]
    latent_risk = (
        1.8 * signals["M"] + 1.5 * signals["V"]
        + 2.0 * signals["D"] + 1.7 * signals["U"]
        + 2.5 * (signals["M"] * signals["V"] + signals["D"] * signals["U"])
        + rng.normal(0.0, 0.25, rows)
    )
    history["failed"] = (latent_risk >= latent_risk.quantile(0.60)).astype(int)
    return history


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    reference, incoming, probabilities = make_batches()
    rules = {
        "age": lambda values: values.between(0, 120),
        "bmi": lambda values: values.between(10, 60),
    }

    calibration_history = make_calibration_history()
    fitted = dfre.calibrate(
        calibration_history,
        label_col="failed",
        cv_folds=3,
        bootstrap=50,
        seed=7,
    )
    calibrated_scores = dfre.score_array(
        calibration_history["M"].to_numpy(),
        calibration_history["V"].to_numpy(),
        calibration_history["D"].to_numpy(),
        calibration_history["U"].to_numpy(),
        n=len(calibration_history),
        lam=fitted.lam,
        gamma=fitted.gamma,
    )
    thresholds = dfre.calibrate_thresholds(
        calibration_history["failed"].to_numpy(), calibrated_scores,
        cost_fp=1.0, cost_fn=5.0,
    )
    policy = to_policy(thresholds)
    print(f"Calibration: lambda={fitted.lam:g}, gamma={fitted.gamma:g}, AUC={fitted.auc:.3f}")
    print(f"Calibrated policy: {policy.to_dict()}")

    pipeline = (
        dfre.DFREBuilder()
        .with_missingness(dfre.MissingnessSignal())
        .with_invalidity(dfre.InvaliditySignal(rules))
        .with_drift(dfre.DriftSignal(
            ["age", "bmi"], categorical_cols=["plan"], statistic="psi", scale=0.25,
        ))
        .with_uncertainty(dfre.UncertaintySignal(method="entropy"))
        .with_extra_signal("freshness", dfre.FreshnessSignal(sla_seconds=3600), weight=0.8)
        .with_extra_signal("volume", dfre.VolumeSignal(expected_rows=len(reference)), weight=0.5)
        .with_extra_signal("schema", dfre.SchemaSignal(["age", "bmi", "plan"]), weight=1.0)
        .with_extra_signal("outliers", dfre.OutlierSignal(["age", "bmi"]), weight=0.5)
        .with_parameters(lam=fitted.lam, gamma=fitted.gamma)
        .with_policy(policy)
        .build()
    )
    last_updated = datetime.now(timezone.utc) - timedelta(minutes=90)
    result = pipeline.evaluate(
        incoming,
        reference,
        probabilities,
        context={
            "freshness": (last_updated,),
            "volume": (len(incoming),),
        },
        metadata={"source": "synthetic_customer_feed", "run": "experiment"},
    )
    print(f"\nIncoming batch: risk={result.risk:.4f} band={result.band}")
    print(f"Action: {result.action}")
    print(f"Base signals: {result.inputs}")
    print(f"Additional signals: {result.extra_signals}")

    signal_values = {
        **{name: result.inputs[name] for name in ("M", "V", "D", "U")},
        **result.extra_signals,
    }
    explanation = dfre.explain(
        signal_values,
        weights=result.weights,
        n=len(incoming),
        lam=fitted.lam,
        gamma=fitted.gamma,
    )
    print(f"\nTop risk driver: {explanation.top_driver}")
    print(explanation.text)
    sensitivity_options = {
        "weights": result.weights,
        "n": len(incoming),
        "lam": fitted.lam,
        "gamma": fitted.gamma,
    }
    gradients = dfre.partial_derivatives(signal_values, **sensitivity_options)
    print("Sensitivity (dR/dsignal):", {key: round(value, 4)
                                        for key, value in gradients.items()})
    print("Tornado analysis:", dfre.tornado(
        signal_values, delta=0.1, **sensitivity_options,
    ))
    print("Elasticity:", dfre.elasticity(signal_values, **sensitivity_options))

    generalized = dfre.score_generalized(
        signal_values, weights=result.weights, n=len(incoming),
        lam=fitted.lam, gamma=fitted.gamma, policy=policy,
    )
    print(f"Generalized scorer: risk={generalized.risk:.4f}, band={generalized.band}")
    print("Signal budget for risk 0.20:",
          dfre.inverse_dfre(0.20, n=len(incoming)))

    model_path = OUTPUT_DIR / "calibrated_model.json"
    model = dfre.DFRE(
        lam=fitted.lam,
        gamma=fitted.gamma,
        policy=policy,
        weights=result.weights,
        history_path=OUTPUT_DIR / "facade_history.jsonl",
        metadata={"model_version": "demo-1"},
    )
    facade_result = model.score_generalized(signal_values, n=len(incoming))
    model.save(model_path)
    restored = dfre.DFRE.load(model_path)
    restored_result = restored.score_generalized(signal_values, n=len(incoming))
    print(f"Saved/reloaded facade: risk={restored_result.risk:.4f} "
          f"(initial={facade_result.risk:.4f})")

    transformed = DFRETransformer(
        lam=fitted.lam, gamma=fitted.gamma, n=len(incoming),
    ).fit_transform(calibration_history[["M", "V", "D", "U"]].to_numpy())
    print(f"scikit-learn transformer output shape: {transformed.shape}")

    accessor_result = incoming.dfre.score(
        reference=reference,
        rules=rules,
        continuous_cols=["age", "bmi"],
        probabilities=probabilities,
        expected_columns=["age", "bmi", "plan"],
        outlier_cols=["age", "bmi"],
        lam=fitted.lam,
        gamma=fitted.gamma,
    )
    print(f"Pandas accessor: risk={accessor_result.risk:.4f}")

    snapshot = dfre.make_snapshot(
        result.inputs["M"], result.inputs["V"], result.inputs["D"],
        result.inputs["U"], len(incoming),
        batch_id="synthetic-incoming-001",
        source="synthetic_customer_feed",
        extra_signals=result.extra_signals,
        metadata={"model_version": "demo-1"},
    )
    history = dfre.HistoryStore(OUTPUT_DIR / "history.jsonl")
    history.append(result, batch_id=snapshot.batch_id)
    alerts = dfre.AlertRouter().add_sink(dfre.file_sink(OUTPUT_DIR / "alerts.jsonl"))
    alerts.dispatch(snapshot, result)
    trend = dfre.TrendDetector().analyze(history.risks())
    print(f"\nHistory summary: {history.stats()}")
    print(f"Trend: {trend.direction}, slope={trend.slope:+.5f}, alarm={trend.alarm}")

    escalation = dfre.EscalationPolicy(policy, consecutive=2, escalate_at="HIGH")
    print("Escalation example:", [escalation.classify(score)
                                  for score in (0.18, 0.18, 0.01, 0.01)])

    report_path = OUTPUT_DIR / "risk_report.html"
    report_path.write_text(
        dfre.generate_html_report(history.load(), title="Customer feed risk"),
        encoding="utf-8",
    )
    print(f"\nReport: {report_path.resolve()}")
    print(f"History and alert logs: {OUTPUT_DIR.resolve()}")
    print("For HTTP serving, install the serve extra and run: dfre serve --help")


if __name__ == "__main__":
    main()