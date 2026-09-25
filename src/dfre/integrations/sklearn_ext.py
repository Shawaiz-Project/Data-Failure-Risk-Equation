"""Scikit-learn compatible transformer producing a DFRE risk feature."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as e:  # pragma: no cover
    raise ImportError("scikit-learn required; pip install dfre[calibration]") from e

from ..core.equation import dfre_array


class DFRETransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that appends a DFRE risk column.

    Expects input ``X`` with columns ``[M, V, D, U]`` (in that order).

    Examples
    --------
    >>> import numpy as np
    >>> from dfre.integrations.sklearn_ext import DFRETransformer
    >>> X = np.array([[0.1, 0.1, 0.1, 0.1], [0.6, 0.6, 0.6, 0.6]])
    >>> DFRETransformer().fit_transform(X).shape
    (2, 5)
    """

    def __init__(self, lam: float = 1.0, gamma: float = 1.0, n: int = 1000,
                 append: bool = True) -> None:
        self.lam = lam
        self.gamma = gamma
        self.n = n
        self.append = append

    def fit(self, X, y=None) -> "DFRETransformer":
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] < 4:
            raise ValueError("X must be 2-D with at least 4 columns (M, V, D, U)")
        M, V, D, U = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        risk = dfre_array(M, V, D, U, n=self.n, lam=self.lam, gamma=self.gamma)
        if not self.append:
            return risk.reshape(-1, 1)
        return np.column_stack([X, risk])

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else \
               ["M", "V", "D", "U"]
        return np.array(base + ["dfre_risk"], dtype=object)
