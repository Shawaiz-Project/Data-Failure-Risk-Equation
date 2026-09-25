import numpy as np
import pandas as pd
import pytest

from dfre.signals.missingness import MissingnessSignal
from dfre.signals.invalidity import InvaliditySignal
from dfre.signals.drift import DriftSignal, psi
from dfre.signals.uncertainty import UncertaintySignal


@pytest.fixture
def clean_batch():
    return pd.DataFrame({
        "age": [25, 30, 45, 50, 60],
        "bmi": [22.0, 24.5, 27.0, 30.0, 32.0],
        "label": ["a", "b", "a", "b", "a"],
    })


@pytest.fixture
def dirty_batch():
    return pd.DataFrame({
        "age": [25, None, 150, 50, -5],
        "bmi": [22.0, 24.5, None, 30.0, 32.0],
        "label": ["a", "z", "a", "b", None],
    })


def test_missingness_clean(clean_batch):
    assert MissingnessSignal().compute(clean_batch) == 0.0


def test_missingness_dirty(dirty_batch):
    M = MissingnessSignal().compute(dirty_batch)
    assert 0 < M < 1


def test_missingness_none_input():
    assert MissingnessSignal().compute(None) == 0.0


def test_invalidity_rules(dirty_batch):
    rules = {
        "age": lambda s: s.between(0, 120),
        "label": lambda s: s.isin(["a", "b"]),
    }
    V = InvaliditySignal(rules).compute(dirty_batch)
    assert 0 < V < 1


def test_invalidity_no_rules(dirty_batch):
    assert InvaliditySignal({}).compute(dirty_batch) == 0.0


def test_drift_psi_identical():
    x = np.random.default_rng(0).normal(0, 1, 1000)
    assert psi(x, x) < 0.05


def test_drift_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(2, 1, 2000)
    assert psi(ref, cur) > 0.5


def test_drift_signal():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"age": rng.normal(45, 10, 1000)})
    shifted = ref.copy()
    shifted["age"] = shifted["age"] + 15
    D = DriftSignal(["age"]).compute(ref, shifted)
    assert D > 0.5


def test_uncertainty_entropy_uniform():
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert UncertaintySignal().compute(probs) == pytest.approx(1.0)


def test_uncertainty_entropy_confident():
    probs = np.array([[0.99, 0.01], [0.99, 0.01]])
    assert UncertaintySignal().compute(probs) < 0.2


def test_uncertainty_margin():
    probs = np.array([[0.9, 0.1], [0.55, 0.45]])
    u = UncertaintySignal("margin").compute(probs)
    assert 0 < u < 1
