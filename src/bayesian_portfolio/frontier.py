"""
Efficient frontier via closed-form Lagrangian optimisation.

Min-variance: w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1)
Max-Sharpe:   w* = Sigma^{-1}(mu-rf) / sum(...)
Target-return: bordered Hessian (n+2)×(n+2) KKT system

References: Markowitz (1952); Merton (1972).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def min_variance_weights(Sigma: np.ndarray) -> np.ndarray:
    r"""
    Global minimum-variance portfolio: w* = Sigma^{-1} 1 / (1^T Sigma^{-1} 1).

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)  SPD covariance.

    Raises
    ------
    ValueError  if system is degenerate (Sigma near-singular).
    """
    Sigma = np.asarray(Sigma, dtype=float)
    ones  = np.ones(Sigma.shape[0])
    v     = np.linalg.solve(Sigma, ones)
    if abs(v.sum()) < 1e-12:
        raise ValueError("min_variance: degenerate — sum(Sigma^{-1} 1) ≈ 0.")
    return v / v.sum()


def max_sharpe_weights(
    Sigma: np.ndarray,
    mu: np.ndarray,
    rf: float = 0.0,
) -> np.ndarray:
    r"""
    Max-Sharpe (tangency) portfolio: z = Sigma^{-1}(mu-rf), w* = z/sum(z).

    Raises if z.sum() ≈ 0 (degenerate KKT).

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)
    mu    : np.ndarray, shape (n,)
    rf    : float
    """
    Sigma = np.asarray(Sigma, dtype=float)
    mu    = np.asarray(mu, dtype=float)
    z     = np.linalg.solve(Sigma, mu - rf)
    if abs(z.sum()) < 1e-12:
        raise ValueError(f"max_sharpe: tangency undefined — z.sum() ≈ 0.")
    return z / z.sum()


def target_return_weights(
    Sigma: np.ndarray,
    mu: np.ndarray,
    mu_target: float,
) -> np.ndarray:
    r"""
    Min-variance portfolio at target return via bordered Hessian KKT:

    .. math::

        \begin{bmatrix} 2\Sigma & \mathbf{1} & \mu \\ \mathbf{1}^\top & 0 & 0 \\
        \mu^\top & 0 & 0 \end{bmatrix}
        \begin{bmatrix} w \\ \lambda_1 \\ \lambda_2 \end{bmatrix}
        = \begin{bmatrix} \mathbf{0} \\ 1 \\ \mu^* \end{bmatrix}

    Parameters
    ----------
    Sigma     : np.ndarray, shape (n, n)
    mu        : np.ndarray, shape (n,)
    mu_target : float

    Raises
    ------
    ValueError              if target is infeasible
    np.linalg.LinAlgError   propagated; caller may skip degenerate points
    """
    Sigma = np.asarray(Sigma, dtype=float)
    mu    = np.asarray(mu, dtype=float)
    n     = len(mu)
    ones  = np.ones(n)

    # All expected returns equal — return constraint is either trivial or infeasible
    if mu.max() - mu.min() < 1e-12:
        if abs(mu_target - mu.mean()) > 1e-8:
            raise ValueError(f"target_return infeasible: all mu={mu.mean():.6f}, target={mu_target:.6f}.")
        return min_variance_weights(Sigma)

    K = np.zeros((n + 2, n + 2))
    K[:n, :n] = 2.0 * Sigma
    K[:n,  n] = ones;      K[n,  :n] = ones
    K[:n, n+1] = mu;       K[n+1, :n] = mu

    rhs      = np.zeros(n + 2)
    rhs[n]   = 1.0
    rhs[n+1] = mu_target

    return np.linalg.solve(K, rhs)[:n]


class EfficientFrontier:
    """
    Efficient frontier via closed-form Lagrangian solutions.

    Parameters
    ----------
    mu    : np.ndarray, shape (n,)
    Sigma : np.ndarray, shape (n, n)  SPD — verified via Cholesky on init.
    rf    : float  risk-free rate for Sharpe computation.
    """

    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, rf: float = 0.0) -> None:
        self.mu    = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.rf    = float(rf)
        self.n     = len(self.mu)
        if self.Sigma.shape != (self.n, self.n):
            raise ValueError(f"Sigma {self.Sigma.shape} inconsistent with mu length {self.n}.")
        try:
            np.linalg.cholesky(self.Sigma)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Sigma not SPD: {exc}") from exc

    def min_variance(self) -> np.ndarray:
        """Global minimum-variance weights."""
        return min_variance_weights(self.Sigma)

    def max_sharpe(self) -> np.ndarray:
        """Max-Sharpe (tangency) weights."""
        return max_sharpe_weights(self.Sigma, self.mu, self.rf)

    def target_return(self, mu_target: float) -> np.ndarray:
        """Min-variance weights for a given target return."""
        return target_return_weights(self.Sigma, self.mu, mu_target)

    def compute_frontier(self, n_points: int = 200) -> pd.DataFrame:
        """
        Sweep target returns from min-variance to max-asset return.

        Returns
        -------
        pd.DataFrame  columns: target_return, portfolio_return, portfolio_std,
                               sharpe_ratio, weights
        """
        w_mv  = self.min_variance()
        r_min = float(w_mv @ self.mu)
        r_max = float(self.mu.max())

        # Slight inset from endpoints avoids near-singular KKT at boundaries
        eps     = (r_max - r_min) * 1e-4
        targets = np.linspace(r_min + eps, r_max - eps, n_points)

        rows = []
        for r_t in targets:
            try:
                w = target_return_weights(self.Sigma, self.mu, float(r_t))
            except (np.linalg.LinAlgError, ValueError):
                continue
            var      = float(w @ self.Sigma @ w)
            std      = float(np.sqrt(max(var, 0.0)))
            port_ret = float(w @ self.mu)
            rows.append({
                "target_return":   r_t,
                "portfolio_return": port_ret,
                "portfolio_std":   std,
                "sharpe_ratio":    (port_ret - self.rf) / (std + 1e-15),
                "weights":         w,
            })
        return pd.DataFrame(rows)

    def two_fund_separation(self, alpha: float) -> np.ndarray:
        r"""
        Any efficient portfolio = alpha*w_tangency + (1-alpha)*w_minvar.

        Weights sum to 1 for any alpha (Merton 1972).
        """
        return float(alpha) * self.max_sharpe() + (1.0 - float(alpha)) * self.min_variance()


class FrontierComparison:
    """
    Compare efficient frontiers across multiple (mu, Sigma) estimators.

    Parameters
    ----------
    estimators : dict[str, tuple[np.ndarray, np.ndarray]]
        name → (mu, Sigma). First entry drawn solid (BL), rest dashed.
    rf : float
    """

    def __init__(
        self,
        estimators: dict[str, tuple[np.ndarray, np.ndarray]],
        rf: float = 0.0,
    ) -> None:
        self.estimators = estimators
        self.rf = float(rf)
        self._frontiers: dict[str, pd.DataFrame] = {}
        self._ef_objects: dict[str, EfficientFrontier] = {}
        for name, (mu, Sigma) in estimators.items():
            ef = EfficientFrontier(mu=mu, Sigma=Sigma, rf=rf)
            self._ef_objects[name] = ef
            self._frontiers[name]  = ef.compute_frontier()

    def plot_frontiers(self, ax: Axes | None = None) -> Axes:
        """
        Plot all frontiers. First entry solid; others dashed.
        Marks max-Sharpe (★) and min-variance (●).
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        names  = list(self.estimators.keys())
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

        for i, (name, frontier) in enumerate(self._frontiers.items()):
            ls = "-" if i == 0 else "--"
            ax.plot(frontier["portfolio_std"], frontier["portfolio_return"],
                    ls=ls, color=colors[i], lw=2, label=name)

            ef = self._ef_objects[name]
            try:
                w_ms = ef.max_sharpe()
                ax.scatter(float(np.sqrt(w_ms @ ef.Sigma @ w_ms)),
                           float(w_ms @ ef.mu),
                           marker="*", s=150, color=colors[i], zorder=5)
            except ValueError:
                pass

            w_mv = ef.min_variance()
            ax.scatter(float(np.sqrt(w_mv @ ef.Sigma @ w_mv)),
                       float(w_mv @ ef.mu),
                       marker="o", s=80, color=colors[i], zorder=5)

        ax.set_xlabel("Portfolio Std Dev")
        ax.set_ylabel("Portfolio Return")
        ax.set_title("Efficient Frontiers: Estimator Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def compare_max_sharpe(self, rf: float | None = None) -> pd.DataFrame:
        """Max-Sharpe weights, return, vol, Sharpe across all estimators."""
        if rf is None:
            rf = self.rf
        rows = {}
        for name, ef in self._ef_objects.items():
            try:
                w = ef.max_sharpe()
            except ValueError:
                continue
            r   = float(w @ ef.mu)
            vol = float(np.sqrt(w @ ef.Sigma @ w))
            row = {"return": r, "volatility": vol, "sharpe": (r - rf) / (vol + 1e-15)}
            for j, wj in enumerate(w):
                row[f"w_{j}"] = float(wj)
            rows[name] = row
        return pd.DataFrame(rows).T
