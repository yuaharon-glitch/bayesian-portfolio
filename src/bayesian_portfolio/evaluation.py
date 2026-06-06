"""
Walk-forward out-of-sample evaluation with Diebold-Mariano statistical tests.

References: Diebold & Mariano (1995) JBES 13(3); Harvey et al. (1997) IJF 13.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from bayesian_portfolio.frontier import max_sharpe_weights, min_variance_weights

RANDOM_SEED: int = 42


def diebold_mariano_test(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    h: int = 1,
) -> dict[str, float]:
    r"""
    Diebold-Mariano test of equal predictive accuracy (H0: d_bar = 0).

    Uses Newey-West HAC variance (Bartlett kernel, h-1 lags) and the
    Harvey-Newbold-Amin small-sample correction. Two-tailed t(T-1) p-value.

    Negative DM* → losses_a < losses_b (A outperforms B).
    Positive DM* → losses_a > losses_b (B outperforms A).

    Parameters
    ----------
    losses_a, losses_b : np.ndarray, shape (T,)
        Per-period loss series. Lower is better (e.g., -r^2).
    h : int
        Forecast horizon; controls NW lag truncation. Default 1.

    Returns
    -------
    dict  keys: dm_statistic, p_value, h, T, d_bar, s2_hac
    """
    losses_a = np.asarray(losses_a, dtype=float)
    losses_b = np.asarray(losses_b, dtype=float)
    d = losses_a - losses_b
    T = len(d)
    if T < 2:
        raise ValueError(f"Need ≥ 2 periods; got T={T}.")

    d_bar  = d.mean()
    s2_hac = float(np.mean((d - d_bar) ** 2))   # gamma_0

    for j in range(1, h):
        gamma_j = float(np.sum((d[j:] - d_bar) * (d[:-j] - d_bar)) / T)
        s2_hac += 2.0 * (1.0 - j / h) * gamma_j  # Bartlett weight

    s2_hac = max(s2_hac, 1e-20)  # prevent sqrt(negative) from numerical noise

    dm_raw  = d_bar / np.sqrt(s2_hac / T)
    # Harvey-Newbold-Amin finite-sample correction
    dm_star = float(dm_raw * np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1) / T) / T))
    p_value = float(2.0 * stats.t.sf(abs(dm_star), df=T - 1))

    return {"dm_statistic": dm_star, "p_value": p_value,
            "h": h, "T": T, "d_bar": float(d_bar), "s2_hac": s2_hac}


@dataclass
class WalkForwardEvaluator:
    """
    Rolling walk-forward backtester for covariance-based portfolio strategies.

    Parameters
    ----------
    train_window : int  e.g. 252 (one year)
    test_window  : int  e.g. 63 (one quarter)
    stride       : int  e.g. 21 (monthly re-estimation)

    Attributes
    ----------
    results_    : per-strategy test-period return arrays, one per fold
    weights_    : per-strategy weight vectors, one per fold
    fold_dates_ : (train_end_date, test_end_date) tuples
    """

    train_window: int
    test_window: int
    stride: int

    results_:    dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)
    weights_:    dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)
    fold_dates_: list[tuple]                 = field(default_factory=list, init=False)
    _n_assets:   int                         = field(default=0, init=False)
    _asset_names: list[str]                  = field(default_factory=list, init=False)

    def run(
        self,
        returns: pd.DataFrame,
        estimators: dict[str, object],
        optimizer: str = "max_sharpe",
        rf: float = 0.0,
    ) -> WalkForwardEvaluator:
        """
        Execute walk-forward evaluation.

        Parameters
        ----------
        returns    : pd.DataFrame, shape (T, n), DatetimeIndex
        estimators : name → covariance estimator class (callable returning a fresh instance)
        optimizer  : 'max_sharpe' | 'min_variance'
        rf         : risk-free rate

        Raises
        ------
        AssertionError  if train period overlaps test period (zero-leakage guard)
        """
        returns = returns.copy()
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        X     = returns.values.astype(float)
        dates = returns.index
        T, n  = X.shape
        self._n_assets   = n
        self._asset_names = list(returns.columns)

        self.results_    = {name: [] for name in estimators}
        self.weights_    = {name: [] for name in estimators}
        self.fold_dates_ = []

        start = 0
        while start + self.train_window + self.test_window <= T:
            train_end = start + self.train_window
            test_end  = start + self.train_window + self.test_window

            # Zero data leakage: test period must strictly follow training period
            assert dates[train_end - 1] < dates[train_end], (
                f"Leakage: train_last={dates[train_end-1]}, test_first={dates[train_end]}"
            )

            X_train  = X[start:train_end]
            X_test   = X[train_end:test_end]
            mu_train = X_train.mean(axis=0)

            for name, factory in estimators.items():
                try:
                    est = factory() if callable(factory) else factory
                    est.fit(X_train)
                    Sigma = est.covariance_
                    w = (max_sharpe_weights(Sigma, mu_train, rf)
                         if optimizer == "max_sharpe"
                         else min_variance_weights(Sigma))
                except (ValueError, np.linalg.LinAlgError) as e:
                    warnings.warn(f"Fold {start}: '{name}' failed ({e}). Using equal weights.")
                    w = np.ones(n) / n

                self.results_[name].append(X_test @ w)
                self.weights_[name].append(w.copy())

            self.fold_dates_.append((dates[train_end - 1], dates[test_end - 1]))
            start += self.stride

        return self

    def portfolio_returns(self, name: str) -> np.ndarray:
        """Concatenated out-of-sample returns for strategy `name`."""
        if name not in self.results_:
            raise KeyError(f"'{name}' not found. Available: {list(self.results_)}")
        return np.concatenate(self.results_[name])

    def summary(self, rf_annual: float = 0.0) -> pd.DataFrame:
        """
        Annualised performance table (252 trading days/year).

        Columns: ann_return, ann_vol, sharpe, max_drawdown, turnover, hit_rate, n_folds, n_obs.
        """
        rows = {}
        for name in self.results_:
            r = self.portfolio_returns(name)
            if len(r) == 0:
                continue

            ann_return = float((1.0 + r.mean()) ** 252 - 1.0)
            ann_vol    = float(r.std(ddof=1) * np.sqrt(252))
            sharpe     = (ann_return - rf_annual) / (ann_vol + 1e-15)

            cum  = np.cumprod(1.0 + r)
            peak = np.maximum.accumulate(cum)
            max_dd = float(((cum - peak) / peak).min())

            ws = self.weights_[name]
            turnover = (float(np.mean([np.abs(ws[i] - ws[i-1]).sum() for i in range(1, len(ws))]))
                        if len(ws) > 1 else float("nan"))

            rows[name] = {
                "ann_return":   ann_return,
                "ann_vol":      ann_vol,
                "sharpe":       sharpe,
                "max_drawdown": max_dd,
                "turnover":     turnover,
                "hit_rate":     float((r > 0).mean()),
                "n_folds":      len(self.results_[name]),
                "n_obs":        len(r),
            }
        return pd.DataFrame(rows).T

    def drawdown_series(self, name: str) -> pd.Series:
        """Cumulative drawdown at each out-of-sample observation (≤ 0)."""
        r    = self.portfolio_returns(name)
        cum  = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        return pd.Series((cum - peak) / peak, name=f"{name}_drawdown")

    def statistical_tests(
        self,
        baseline: str = "sample",
        h: int = 1,
        loss: str = "squared_return",
    ) -> pd.DataFrame:
        """
        DM tests comparing each strategy to `baseline`.

        A positive DM statistic means the alternative has lower loss (outperforms).
        Never claim outperformance without p_value < 0.05.

        Parameters
        ----------
        baseline : str
        h        : int  NW lag truncation
        loss     : 'squared_return' | 'absolute_return'
        """
        if baseline not in self.results_:
            raise KeyError(f"'{baseline}' not in results_. Available: {list(self.results_)}")

        def _loss(r: np.ndarray) -> np.ndarray:
            if loss == "squared_return":
                return -(r ** 2)
            if loss == "absolute_return":
                return -np.abs(r)
            raise ValueError(f"Unknown loss '{loss}'.")

        L_base = _loss(self.portfolio_returns(baseline))
        rows   = {}
        for name in self.results_:
            if name == baseline:
                continue
            r_alt   = self.portfolio_returns(name)
            min_len = min(len(L_base), len(r_alt))
            try:
                res = diebold_mariano_test(L_base[:min_len], _loss(r_alt[:min_len]), h=h)
                rows[name] = {**res, "significant_5pct": res["p_value"] < 0.05}
            except (ValueError, ZeroDivisionError):
                rows[name] = {"dm_statistic": float("nan"), "p_value": float("nan"),
                              "d_bar": float("nan"), "h": h, "T": min_len,
                              "significant_5pct": False}
        return pd.DataFrame(rows).T
