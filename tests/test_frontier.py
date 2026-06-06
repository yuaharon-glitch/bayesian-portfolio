"""Tests for bayesian_portfolio.frontier — closed-form efficient frontier."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_portfolio.frontier import (
    EfficientFrontier,
    FrontierComparison,
    max_sharpe_weights,
    min_variance_weights,
    target_return_weights,
)

RANDOM_SEED = 42


def _make_sigma_mu(n: int = 10, seed: int = RANDOM_SEED):
    """Random SPD covariance and strictly positive returns (ensures z.sum() > 0)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + 0.05 * np.eye(n)
    mu = np.abs(rng.standard_normal(n)) * 0.005 + 0.003   # strictly positive
    return Sigma, mu


def test_min_variance_sums_to_one():
    """Min-variance weights sum to 1 within 1e-8."""
    Sigma, mu = _make_sigma_mu()
    w = min_variance_weights(Sigma)
    assert abs(w.sum() - 1.0) < 1e-8, f"w.sum()={w.sum():.12f} != 1"


def test_min_variance_lower_than_equal_weight():
    """Min-variance portfolio has lower variance than the equal-weight portfolio."""
    Sigma, mu = _make_sigma_mu()
    n = len(mu)
    w_mv = min_variance_weights(Sigma)
    w_eq = np.ones(n) / n
    var_mv = float(w_mv @ Sigma @ w_mv)
    var_eq = float(w_eq @ Sigma @ w_eq)
    assert var_mv < var_eq, (
        f"Min-var variance ({var_mv:.6e}) not less than equal-weight ({var_eq:.6e})"
    )


def test_min_variance_is_global_minimum():
    """Min-variance return lies at or below sample mean portfolio return."""
    Sigma, mu = _make_sigma_mu()
    n = len(mu)
    w_mv = min_variance_weights(Sigma)
    # Perturb weights slightly; variance should be >= min-var
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(20):
        delta = rng.standard_normal(n) * 0.01
        w_perturbed = w_mv + delta
        w_perturbed /= w_perturbed.sum()
        var_perturbed = float(w_perturbed @ Sigma @ w_perturbed)
        assert var_perturbed >= float(w_mv @ Sigma @ w_mv) - 1e-10


def test_max_sharpe_sums_to_one():
    """Max-Sharpe weights sum to 1 within 1e-8."""
    Sigma, mu = _make_sigma_mu()
    w = max_sharpe_weights(Sigma, mu, rf=0.0)
    assert abs(w.sum() - 1.0) < 1e-8, f"w.sum()={w.sum():.12f} != 1"


def test_max_sharpe_rf_nonzero():
    """Max_sharpe_weights still sums to 1 with nonzero rf."""
    Sigma, mu = _make_sigma_mu()
    rf = 0.002
    w = max_sharpe_weights(Sigma, mu, rf=rf)
    assert abs(w.sum() - 1.0) < 1e-8


def test_max_sharpe_higher_sharpe_than_equal_weight():
    """Tangency portfolio should have higher Sharpe than equal-weight
    in standard cases (positive excess returns)."""
    Sigma, mu = _make_sigma_mu()
    n = len(mu)
    rf = 0.0
    w_ms = max_sharpe_weights(Sigma, mu, rf=rf)
    w_eq = np.ones(n) / n

    def sharpe(w):
        r = float(w @ mu)
        vol = float(np.sqrt(w @ Sigma @ w))
        return (r - rf) / vol

    assert sharpe(w_ms) >= sharpe(w_eq) - 1e-8, (
        f"Max-Sharpe ({sharpe(w_ms):.4f}) should be >= equal-weight ({sharpe(w_eq):.4f})"
    )


def test_target_return_achieves_target():
    """Target_return_weights achieves mu_target within 1e-6."""
    Sigma, mu = _make_sigma_mu()
    w_mv = min_variance_weights(Sigma)
    r_min = float(w_mv @ mu)
    r_max = float(mu.max())
    mu_target = (r_min + r_max) / 2.0

    w = target_return_weights(Sigma, mu, mu_target)
    achieved = float(w @ mu)
    assert abs(achieved - mu_target) < 1e-6, (
        f"Target {mu_target:.8f} not achieved: got {achieved:.8f}, "
        f"deviation = {abs(achieved - mu_target):.2e}"
    )


def test_target_return_weights_sum_to_one():
    """Target_return weights sum to 1 within 1e-8."""
    Sigma, mu = _make_sigma_mu()
    w_mv = min_variance_weights(Sigma)
    r_min = float(w_mv @ mu)
    r_max = float(mu.max())
    mu_target = (r_min + r_max) / 2.0

    w = target_return_weights(Sigma, mu, mu_target)
    assert abs(w.sum() - 1.0) < 1e-8, (
        f"Weights sum to {w.sum():.12f}, not 1.0"
    )


def test_target_min_variance_return_matches():
    """Targeting the min-variance return should recover near min-var weights."""
    Sigma, mu = _make_sigma_mu()
    w_mv = min_variance_weights(Sigma)
    r_mv = float(w_mv @ mu)

    w = target_return_weights(Sigma, mu, r_mv)
    assert abs(w @ mu - r_mv) < 1e-6
    # Variance of targeted portfolio should equal min-var variance
    var_target = float(w @ Sigma @ w)
    var_mv = float(w_mv @ Sigma @ w_mv)
    assert abs(var_target - var_mv) < 1e-8, (
        f"Target at r_mv: var={var_target:.8e}, min-var={var_mv:.8e}"
    )


def test_ef_compute_frontier_returns_dataframe():
    """compute_frontier() returns a non-empty DataFrame with required columns."""
    Sigma, mu = _make_sigma_mu()
    ef = EfficientFrontier(mu=mu, Sigma=Sigma)
    df = ef.compute_frontier(n_points=50)
    required = {"target_return", "portfolio_return", "portfolio_std", "sharpe_ratio", "weights"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    assert len(df) > 0, "Frontier DataFrame is empty"


def test_ef_two_fund_separation_sums_to_one():
    """two_fund_separation(alpha) weights sum to 1 for any alpha."""
    Sigma, mu = _make_sigma_mu()
    ef = EfficientFrontier(mu=mu, Sigma=Sigma)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        w = ef.two_fund_separation(alpha)
        assert abs(w.sum() - 1.0) < 1e-10, (
            f"two_fund_separation({alpha}): w.sum()={w.sum():.12f}"
        )


def test_ef_raises_on_non_spd_sigma():
    """EfficientFrontier raises ValueError for non-SPD Sigma."""
    mu = np.array([0.01, 0.02])
    Sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # not PD (eigenvalue -1)
    with pytest.raises(ValueError):
        EfficientFrontier(mu=mu, Sigma=Sigma)


def test_frontier_comparison_shape():
    """FrontierComparison.compare_max_sharpe returns correct shape."""
    Sigma, mu = _make_sigma_mu(n=5)
    estimators = {
        "BL": (mu, Sigma),
        "Sample": (mu * 0.9, Sigma * 1.1),
    }
    fc = FrontierComparison(estimators)
    df = fc.compare_max_sharpe()
    assert "return" in df.columns
    assert "volatility" in df.columns
    assert "sharpe" in df.columns
    assert len(df) == 2
