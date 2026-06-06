"""Tests for bayesian_portfolio.priors — Normal-Wishart conjugate prior/posterior."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_portfolio.priors import NormalWishartPosterior, NormalWishartPrior

RANDOM_SEED = 42


def _make_returns(T: int = 120, n: int = 5, seed: int = RANDOM_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.01


def _make_prior(
    n: int = 5,
    kappa_0: float = 5.0,
    nu_0: float | None = None,
    mu_0: np.ndarray | None = None,
) -> NormalWishartPrior:
    if nu_0 is None:
        nu_0 = float(n + 2)
    if mu_0 is None:
        mu_0 = np.zeros(n)
    return NormalWishartPrior(mu_0=mu_0, kappa_0=kappa_0, nu_0=nu_0, Psi_0=np.eye(n))


def test_convex_combination_equal_weights():
    """When kappa_0 == T, posterior mean is the midpoint of prior mean and sample mean."""
    n, T = 5, 120
    X = _make_returns(T, n)
    x_bar = X.mean(axis=0)

    prior = _make_prior(n, kappa_0=float(T))
    post = NormalWishartPosterior.update(prior, X)

    expected = (prior.mu_0 + x_bar) / 2.0
    assert np.allclose(post.mu_n, expected, atol=1e-12), (
        f"Expected mu_n = (mu_0 + x_bar)/2, max deviation = "
        f"{np.abs(post.mu_n - expected).max():.2e}"
    )


def test_kappa0_near_zero_gives_sample_mean():
    """With a near-flat prior (kappa_0 ≈ 0), posterior mean ≈ sample mean."""
    n, T = 5, 120
    X = _make_returns(T, n)
    x_bar = X.mean(axis=0)

    prior = _make_prior(n, kappa_0=1e-12)
    post = NormalWishartPosterior.update(prior, X)

    assert np.allclose(post.mu_n, x_bar, atol=1e-6), (
        f"Expected mu_n ≈ x_bar (kappa_0→0), max deviation = "
        f"{np.abs(post.mu_n - x_bar).max():.2e}"
    )


def test_kappa0_large_gives_prior_mean():
    """With a very strong prior (kappa_0 >> T), posterior mean ≈ mu_0."""
    n, T = 5, 120
    rng = np.random.default_rng(RANDOM_SEED)
    mu_0 = rng.standard_normal(n) * 0.05
    X = _make_returns(T, n)

    prior = _make_prior(n, kappa_0=1e12, mu_0=mu_0)
    post = NormalWishartPosterior.update(prior, X)

    assert np.allclose(post.mu_n, mu_0, atol=1e-4), (
        f"Expected mu_n ≈ mu_0 (kappa_0→∞), max deviation = "
        f"{np.abs(post.mu_n - mu_0).max():.2e}"
    )


def test_posterior_covariance_is_spd():
    """Psi_n and posterior_covariance() pass the Cholesky SPD test."""
    n, T = 8, 200
    X = _make_returns(T, n)
    prior = _make_prior(n)
    post = NormalWishartPosterior.update(prior, X)

    try:
        np.linalg.cholesky(post.Psi_n)
    except np.linalg.LinAlgError as exc:
        pytest.fail(f"Psi_n is not SPD: {exc}")

    cov = post.posterior_covariance()
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        pytest.fail(f"posterior_covariance() is not SPD: {exc}")


def test_prior_strength_ratio_bounds():
    """prior_strength_ratio() should be in (0, 1)."""
    n, T = 5, 100
    X = _make_returns(T, n)
    prior = _make_prior(n, kappa_0=10.0)
    post = NormalWishartPosterior.update(prior, X)
    ratio = post.prior_strength_ratio()
    assert 0.0 < ratio < 1.0, f"prior_strength_ratio={ratio} not in (0, 1)"


def test_effective_sample_size():
    """effective_sample_size() should equal kappa_0 + T."""
    n, T = 5, 100
    kappa_0 = 7.5
    X = _make_returns(T, n)
    prior = _make_prior(n, kappa_0=kappa_0)
    post = NormalWishartPosterior.update(prior, X)
    assert abs(post.effective_sample_size() - (kappa_0 + T)) < 1e-12


def test_predictive_covariance_is_spd():
    """predictive_covariance() must be SPD."""
    n, T = 5, 100
    X = _make_returns(T, n)
    prior = _make_prior(n)
    post = NormalWishartPosterior.update(prior, X)
    pred_cov = post.predictive_covariance()
    try:
        np.linalg.cholesky(pred_cov)
    except np.linalg.LinAlgError as exc:
        pytest.fail(f"predictive_covariance() is not SPD: {exc}")


def test_prior_mean_of_covariance():
    """prior_mean_of_covariance() = Psi_0 / (nu_0 - n - 1)."""
    n = 4
    nu_0 = float(n + 3)
    prior = NormalWishartPrior(
        mu_0=np.zeros(n), kappa_0=1.0, nu_0=nu_0, Psi_0=np.eye(n)
    )
    expected = np.eye(n) / (nu_0 - n - 1)
    assert np.allclose(prior.prior_mean_of_covariance(), expected, atol=1e-14)
