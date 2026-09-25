"""Tests for everything added in 0.2.0 (numpy/pandas only)."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

import dfre
from dfre.core.equation import dfre_generalized, inverse_dfre
from dfre.core.sensitivity import partial_derivatives, tornado, elasticity
from dfre.core.errors import InvalidInputError, InvalidParameterError
from dfre.signals import (
    FreshnessSignal, VolumeSignal, SchemaSignal, OutlierSignal, jensen_shannon,
)
from dfre.policy.escalation import EscalationPolicy
from dfre.monitoring.history import HistoryStore
from dfre.monitoring.trends import TrendDetector
from dfre.monitoring.alerts import AlertRouter, file_sink
from dfre.report.html import generate_html_report


# ---- generalized equation -------------------------------------------------

def test_generalized_zero():
    assert dfre_generalized({"M": 0.0, "V": 0.0}) == 0.0


def test_generalized_bounded_and_monotonic():
    r1 = dfre_generalized({"M": 0.2, "F": 0.1})
    r2 = dfre_generalized({"M": 0.2, "F": 0.9})
    assert 0 <= r1 < r2 < 1


def test_generalized_weights():
    plain = dfre_generalized({"M": 0.5, "V": 0.5})
    heavy_m = dfre_generalized({"M": 0.5, "V": 0.5}, weights={"M": 3.0, "V": 1.0})
    assert heavy_m != plain
    assert heavy_m > 0


def test_generalized_validation():
    with pytest.raises(InvalidInputError):
        dfre_generalized({})
    with pytest.raises(InvalidInputError):
        dfre_generalized({"M": 2.0})
    with pytest.raises(InvalidParameterError):
        dfre_generalized({"M": 0.1}, weights={"M": -1})


def test_inverse_dfre_roundtrip():
    budget = inverse_dfre(0.05, n=1000)
    r = dfre_generalized({"M": budget / 2, "V": budget / 2}, n=1000,
                         lam=0, gamma=0)
    assert r == pytest.approx(0.05, abs=1e-6)


# ---- explain & sensitivity -------------------------------------------------

def test_explain_top_driver():
    e = dfre.explain({"M": 0.6, "V": 0.1, "D": 0.1, "U": 0.1})
    assert e.top_driver == "M"
    assert sum(e.shares.values()) == pytest.approx(1.0)
    assert "M" in e.text


def test_explain_all_zero():
    e = dfre.explain({"M": 0.0, "V": 0.0})
    assert e.top_driver is None


def test_partial_derivatives_positive_and_order():
    grads = partial_derivatives({"M": 0.5, "V": 0.1}, n=1000)
    assert all(g > 0 for g in grads.values())
    # V sits next to the larger M, so dR/dV > dR/dM through the interaction terms
    assert grads["V"] > grads["M"]
    assert all(math.isfinite(g) for g in grads.values())


def test_tornado_and_elasticity():
    t = tornado({"M": 0.5, "V": 0.2}, delta=0.1)
    assert set(t) == {"M", "V"}
    assert t["M"]["risk_plus"] > t["M"]["risk_minus"]
    el = elasticity({"M": 0.5, "V": 0.2})
    assert all(math.isfinite(v) for v in el.values())


# ---- new signals ------------------------------------------------------------

def test_freshness():
    now = datetime.now(timezone.utc)
    sig = FreshnessSignal(sla_seconds=3600)
    assert sig.compute(now, now=now) == 0.0
    assert sig.compute(now - timedelta(seconds=1800), now=now) == pytest.approx(0.5)
    assert sig.compute(now - timedelta(seconds=7200), now=now) == 1.0


def test_volume():
    sig = VolumeSignal(expected_rows=1000)
    assert sig.compute(1000) == 0.0
    assert sig.compute(0) == 1.0
    assert 0.5 < sig.compute(500) < 0.8


def test_schema():
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [3]})
    sig = SchemaSignal(["a", "b", "c"])
    s = sig.compute(df)
    assert 0 < s <= 1
    report = sig.report(df)
    assert report["checks"]["present:c"] is False
    assert report["checks"]["no_extra_columns"] is False


def test_outliers():
    x = np.array([1.0, 1.1, 0.9, 1.0, 50.0]).reshape(-1, 1)
    assert OutlierSignal().compute(x) == pytest.approx(0.2)


def test_jensen_shannon_identical():
    a = pd.Series(["x", "y", "x", "y"])
    assert jensen_shannon(a, a) == pytest.approx(0.0)


# ---- escalation -------------------------------------------------------------

def test_escalation_after_consecutive_breaches():
    ep = EscalationPolicy(consecutive=2, escalate_at="HIGH")
    assert ep.classify(0.5) == "HIGH"        # 0.5 is in [0.45, 0.70) -> HIGH
    assert ep.classify(0.5) == "CRITICAL"    # 2nd consecutive HIGH -> escalated


def test_escalation_hysteresis_cooldown():
    ep = EscalationPolicy(consecutive=2, escalate_at="HIGH", cooldown=2)
    ep.classify(0.5)
    ep.classify(0.5)  # elevated to CRITICAL
    assert ep.classify(0.1) == "CRITICAL"   # held (hysteresis)
    assert ep.classify(0.1) == "LOW"        # released after cooldown


# ---- history & trends -------------------------------------------------------

def test_history_store(tmp_path):
    store = HistoryStore(tmp_path / "h.jsonl")
    store.append(dfre.score(0.1, 0.1, 0.1, 0.1), batch_id="b1")
    store.append(dfre.score(0.6, 0.6, 0.6, 0.6), batch_id="b2")
    stats = store.stats()
    assert stats["count"] == 2
    assert stats["min"] < stats["max"]
    assert len(store.risks()) == 2


def test_trend_detector_stable_and_rising():
    assert TrendDetector().analyze([0.1] * 10).direction == "stable"
    rising = TrendDetector().analyze([0.05 * i for i in range(1, 15)])
    assert rising.direction == "increasing"
    assert len(rising.forecast) == 3


def test_file_sink(tmp_path):
    from dfre.monitoring.snapshot import make_snapshot
    target = tmp_path / "alerts.jsonl"
    router = AlertRouter(sinks=[file_sink(target)])
    result = dfre.score(0.9, 0.9, 0.9, 0.9)
    snap = make_snapshot(0.9, 0.9, 0.9, 0.9, 100, batch_id="b9")
    payload = router.dispatch(snap, result)
    assert payload is not None
    lines = target.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["batch_id"] == "b9"


# ---- report -----------------------------------------------------------------

def test_html_report(tmp_path):
    store = HistoryStore(tmp_path / "h.jsonl")
    store.append(dfre.score(0.1, 0.1, 0.1, 0.1), batch_id="b1")
    store.append(dfre.score(0.8, 0.7, 0.7, 0.8, n=10), batch_id="b2")
    html = generate_html_report(store.load())
    assert "<svg" in html
    assert "b1" in html and "b2" in html
    assert "CRITICAL" in html
    out = tmp_path / "report.html"
    out.write_text(html)
    assert out.stat().st_size > 2000


# ---- facade & generalized scoring ------------------------------------------

def test_facade_generalized_and_history(tmp_path):
    model = dfre.DFRE(history_path=tmp_path / "hist.jsonl")
    r = model.score_generalized({"M": 0.3, "V": 0.2, "freshness": 0.4})
    assert 0 <= r.risk < 1
    assert r.extra_signals == {"freshness": 0.4}
    assert model.history().stats()["count"] == 1


def test_facade_save_load_roundtrip(tmp_path):
    model = dfre.DFRE(lam=2.0, gamma=0.5, weights={"M": 2.0})
    p = tmp_path / "model.json"
    model.save(p)
    loaded = dfre.DFRE.load(p)
    assert loaded.lam == 2.0 and loaded.gamma == 0.5
    assert loaded.weights == {"M": 2.0}


def test_score_generalized_one_liner():
    r = dfre.score_generalized({"M": 0.5, "V": 0.5, "D": 0.5, "U": 0.5})
    assert 0 <= r.risk < 1
    assert r.band in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


# ---- builder with extras ----------------------------------------------------

def test_builder_with_extra_signals():
    pipeline = (
        dfre.DFREBuilder()
        .with_extra_signal("outliers", OutlierSignal(), weight=1.0)
        .build()
    )
    batch = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0, 99.0]})
    result = pipeline.evaluate(batch)
    assert "outliers" in result.extra_signals
    assert 0 <= result.risk < 1


def test_pandas_accessor_with_schema():
    import dfre.integrations.pandas_ext  # noqa: F401
    df = pd.DataFrame({"a": [1, 2, None], "b": [3, 4, 5]})
    result = df.dfre.score(expected_columns=["a", "b", "c"], outlier_cols=["b"])
    assert "schema" in result.extra_signals
    assert "outliers" in result.extra_signals


# ---- calibration extras -----------------------------------------------------

def test_calibrate_cv_and_bootstrap():
    rng = np.random.default_rng(0)
    n = 60
    history = pd.DataFrame({
        "M": rng.random(n), "V": rng.random(n),
        "D": rng.random(n), "U": rng.random(n),
    })
    from dfre.core.equation import dfre_array as _dfre_array
    base = _dfre_array(
        history["M"], history["V"], history["D"], history["U"], n=1000)
    history["failed"] = (base > np.median(base)).astype(int)
    result = dfre.calibrate(history, label_col="failed",
                            lam_grid=(0.0, 1.0), gamma_grid=(0.0, 1.0),
                            cv_folds=3, bootstrap=50)
    assert 0.5 <= result.auc <= 1.0
    assert result.auc_ci is not None
    assert result.cv_scores


# ---- classic API regression -------------------------------------------------

def test_classic_score_unchanged_shape():
    r = dfre.score(0.08, 0.04, 0.12, 0.20)
    assert r.band == "LOW"
    assert 0 <= r.risk < 1


def test_cli_score(capsys):
    from dfre.cli import main
    rc = main(["score", "--missing", "0.1", "--invalid", "0.1",
               "--drift", "0.1", "--uncertainty", "0.1"])
    assert rc == 0
    assert "band=LOW" in capsys.readouterr().out


def test_cli_explain(capsys):
    from dfre.cli import main
    rc = main(["explain", "--signal", "M=0.5", "--signal", "V=0.1"])
    assert rc == 0
    assert "driven mainly by M" in capsys.readouterr().out
