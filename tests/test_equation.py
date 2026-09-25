import math

import numpy as np
import pytest

from dfre.core.equation import dfre, dfre_array
from dfre.core.errors import InvalidInputError, InvalidParameterError


def test_zero_inputs_zero_risk():
    assert dfre(0, 0, 0, 0) == pytest.approx(0.0)


def test_bounded_output():
    r = dfre(1, 1, 1, 1, n=1, lam=10, gamma=10)
    assert 0 <= r < 1


def test_monotonic_in_each_signal():
    base = dfre(0.1, 0.1, 0.1, 0.1)
    for field in range(4):
        args = [0.1, 0.1, 0.1, 0.1]
        args[field] = 0.5
        assert dfre(*args) > base


def test_interaction_amplifies():
    flat = dfre(0.3, 0.3, 0.3, 0.3, lam=0, gamma=0)
    rich = dfre(0.3, 0.3, 0.3, 0.3, lam=1, gamma=1)
    assert rich > flat


def test_sample_size_reduces_risk():
    small = dfre(0.3, 0.3, 0.3, 0.3, n=10)
    large = dfre(0.3, 0.3, 0.3, 0.3, n=100000)
    assert small > large


def test_components_sum():
    c = dfre(0.2, 0.3, 0.1, 0.4, lam=1, gamma=1, return_components=True)
    total = c["main"] + c["pairwise"] + c["four_way"]
    assert c["intensity"] == pytest.approx(total / (1 + math.sqrt(1000)))


def test_invalid_input_raises():
    with pytest.raises(InvalidInputError):
        dfre(1.5, 0, 0, 0)
    with pytest.raises(InvalidInputError):
        dfre(-0.1, 0, 0, 0)


def test_invalid_params_raise():
    with pytest.raises(InvalidParameterError):
        dfre(0, 0, 0, 0, n=0)
    with pytest.raises(InvalidParameterError):
        dfre(0, 0, 0, 0, lam=-1)


def test_array_matches_scalar():
    M = np.array([0.1, 0.5, 0.9])
    V = np.array([0.2, 0.4, 0.1])
    D = np.array([0.3, 0.1, 0.0])
    U = np.array([0.4, 0.3, 0.2])
    arr = dfre_array(M, V, D, U, n=1000)
    for i in range(3):
        assert arr[i] == pytest.approx(dfre(M[i], V[i], D[i], U[i]))
