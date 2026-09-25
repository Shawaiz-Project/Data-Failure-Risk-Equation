"""Calibrate λ and γ from historical batches, with optional CV and bootstrap CI."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..core.equation import dfre_array
from ..core.errors import CalibrationError


@dataclass
class CalibrationResult:
    lam: float
    gamma: float
    auc: float
    grid: list
    auc_ci: Optional[tuple] = None          # bootstrap (low, high) for best (λ, γ)
    cv_scores: Dict[str, list] = field(default_factory=dict)  # "λ,γ" -> fold AUCs


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Rank-based AUC with no sklearn dependency."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise CalibrationError("Need both classes present to compute AUC")
    ranks = y_score.argsort().argsort() + 1
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def _scores(history, lam: float, gamma: float, n: int) -> np.ndarray:
    return dfre_array(
        history["M"].to_numpy(), history["V"].to_numpy(),
        history["D"].to_numpy(), history["U"].to_numpy(),
        n=n, lam=lam, gamma=gamma,
    )


def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray,
                   n_boot: int, seed: int) -> Optional[tuple]:
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    size = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, size, size)
        try:
            aucs.append(_roc_auc(y_true[idx], y_score[idx]))
        except CalibrationError:
            continue
    if len(aucs) < 10:
        return None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return (float(lo), float(hi))


def calibrate(
    history,
    *,
    lam_grid: Iterable[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    gamma_grid: Iterable[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    label_col: str = "failed",
    n: Optional[int] = None,
    cv_folds: int = 1,
    bootstrap: int = 0,
    seed: int = 42,
) -> CalibrationResult:
    """Search (λ, γ) maximizing AUC on historical batches.

    Parameters
    ----------
    history : pandas.DataFrame
        Must contain columns M, V, D, U and the label column.
    cv_folds : int, default 1
        If > 1, selects by mean k-fold AUC and reports per-fold scores.
    bootstrap : int, default 0
        If > 0, computes a bootstrap 95% CI for the winning AUC.
    seed : int
        RNG seed for CV splitting and bootstrapping.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for calibration; pip install dfre[pandas]") from e

    required = {"M", "V", "D", "U", label_col}
    missing = required - set(history.columns)
    if missing:
        raise CalibrationError(f"history missing columns: {sorted(missing)}")

    y_true = history[label_col].to_numpy(dtype=int)
    N = n if n is not None else len(history)
    rng = np.random.default_rng(seed)
    fold_ids = (
        rng.permutation(len(history)) % cv_folds if cv_folds > 1 else None
    )

    grid_results: List[dict] = []
    cv_scores: Dict[str, list] = {}
    best = None

    for lam, gamma in product(lam_grid, gamma_grid):
        key = f"{lam},{gamma}"
        scores = _scores(history, lam, gamma, N)

        if cv_folds > 1 and fold_ids is not None:
            fold_aucs: List[float] = []
            for f in range(cv_folds):
                test = fold_ids == f
                try:
                    fold_aucs.append(_roc_auc(y_true[test], scores[test]))
                except CalibrationError:
                    fold_aucs.append(float("nan"))
            cv_scores[key] = fold_aucs
            valid = [a for a in fold_aucs if a == a]
            auc = float(np.mean(valid)) if valid else float("nan")
        else:
            try:
                auc = _roc_auc(y_true, scores)
            except CalibrationError:
                auc = float("nan")

        grid_results.append({"lam": lam, "gamma": gamma, "auc": auc})
        if best is None or (auc == auc and auc > best["auc"]):
            best = {"lam": lam, "gamma": gamma, "auc": auc}

    if best is None or best["auc"] != best["auc"]:
        raise CalibrationError("Calibration failed: no valid (λ, γ) pair found")

    auc_ci = None
    if bootstrap > 0:
        auc_ci = _bootstrap_auc(
            y_true, _scores(history, best["lam"], best["gamma"], N), bootstrap, seed
        )

    return CalibrationResult(
        lam=float(best["lam"]),
        gamma=float(best["gamma"]),
        auc=float(best["auc"]),
        grid=grid_results,
        auc_ci=auc_ci,
        cv_scores=cv_scores,
    )
