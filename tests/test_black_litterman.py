"""Tests for bayesian_portfolio.black_litterman — Black-Litterman model."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_portfolio.black_litterman import BlackLitterman, implied_returns

RANDOM_SEED = 42


def _make_3asset_bl(delta: float = 2.5, tau: float = 0.05) -> BlackLitterman:
    """3-asset BL model used in multiple tests."""
    Sigma = np.array(
        [
            [0.04, 0.010, 0.005],
            [0.01, 0.090, 0.020],
            [0.005, 0.020, 0.160],
        ]
    )
    w_mkt = np.array([0.50, 0.30, 0.20])
    return BlackLitterman(Sigma=Sigma, w_mkt=w_mkt, delta=delta, tau=tau)


def test_no_views_gives_equilibrium():
    """With zero views, posterior mean equals equilibrium Pi exactly."""
    bl = _make_3asset_bl()
    n = bl.n
    mu_bl, Sigma_bl = bl.posterior(
        P=np.zeros((0, n)),
        Q=np.zeros(0),
        Omega=np.zeros((0, 0)),
    )
    assert np.allclose(mu_bl, bl.Pi, atol=1e-12), (
        f"No-view posterior mean deviates from Pi: max={np.abs(mu_bl - bl.Pi).max():.2e}"
    )


def test_no_views_sigma_bl_is_sigma_plus_tau_sigma():
    """With no views, Sigma_BL = Sigma + tau*Sigma = (1+tau)*Sigma."""
    bl = _make_3asset_bl()
    n = bl.n
    _, Sigma_bl = bl.posterior(
        P=np.zeros((0, n)),
        Q=np.zeros(0),
        Omega=np.zeros((0, 0)),
    )
    expected = (1.0 + bl.tau) * bl.Sigma
    assert np.allclose(Sigma_bl, expected, atol=1e-12), (
        "No-view Sigma_BL != (1+tau)*Sigma"
    )


def test_two_asset_one_view_known_properties():
    """BL posterior blends Pi and Q; bullish relative view shifts mu_BL."""
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w_mkt = np.array([0.60, 0.40])
    bl = BlackLitterman(Sigma=Sigma, w_mkt=w_mkt, delta=2.5, tau=0.05)

    # View: asset 0 outperforms asset 1 by 5%
    P = np.array([[1.0, -1.0]])
    Q = np.array([0.05])
    Omega = bl.default_omega(P)

    mu_bl, Sigma_bl = bl.posterior(P, Q, Omega)

    assert mu_bl.shape == (2,), f"Expected shape (2,), got {mu_bl.shape}"

    # Bullish view on asset 0 vs asset 1 should pull the relative spread toward Q
    relative_pi = bl.Pi[0] - bl.Pi[1]
    relative_bl = mu_bl[0] - mu_bl[1]
    assert relative_bl > relative_pi, (
        f"BL relative return ({relative_bl:.4f}) should exceed Pi relative "
        f"({relative_pi:.4f}) for a bullish view Q={Q[0]}"
    )

    assert mu_bl.shape == bl.Pi.shape


def test_single_full_view_collapses_to_Q():
    """With near-zero Omega, posterior mean approaches view Q."""
    Sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w_mkt = np.array([0.60, 0.40])
    bl = BlackLitterman(Sigma=Sigma, w_mkt=w_mkt, delta=2.5, tau=0.05)

    P = np.eye(2)
    Q = np.array([0.10, 0.05])
    Omega = 1e-10 * np.eye(2)

    mu_bl, _ = bl.posterior(P, Q, Omega)

    assert np.allclose(mu_bl, Q, atol=1e-4), (
        f"Near-certain view: mu_BL={mu_bl} should be close to Q={Q}"
    )


def test_implied_returns_round_trip():
    """Recover w_mkt from Pi via solve(Sigma, Pi/delta); must match exactly."""
    bl = _make_3asset_bl()
    Pi = bl.Pi
    Sigma = bl.Sigma
    delta = bl.delta
    w_mkt = bl.w_mkt

    z = np.linalg.solve(Sigma, Pi / delta)
    w_recovered = z / z.sum()

    assert np.allclose(w_recovered, w_mkt, atol=1e-10), (
        f"Round-trip failed: max deviation = {np.abs(w_recovered - w_mkt).max():.2e}"
    )


def test_round_trip_random_portfolio():
    """Round-trip holds for a random SPD Sigma and w_mkt."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 10
    A = rng.standard_normal((n, n))
    Sigma = A @ A.T / n + 0.01 * np.eye(n)
    w_mkt = np.abs(rng.dirichlet(np.ones(n)))   # random weights summing to 1

    delta = 2.5
    Pi = implied_returns(Sigma, w_mkt, delta)

    z = np.linalg.solve(Sigma, Pi / delta)
    w_recovered = z / z.sum()

    assert np.allclose(w_recovered, w_mkt, atol=1e-10), (
        f"Round-trip failed (random): max={np.abs(w_recovered - w_mkt).max():.2e}"
    )


def test_posterior_mu_bl_shape():
    """posterior() returns correctly shaped arrays."""
    bl = _make_3asset_bl()
    P = np.array([[1.0, -0.5, -0.5]])
    Q = np.array([0.02])
    Omega = bl.default_omega(P)
    mu_bl, Sigma_bl = bl.posterior(P, Q, Omega)
    assert mu_bl.shape == (3,)
    assert Sigma_bl.shape == (3, 3)


def test_posterior_sigma_bl_spd():
    """Posterior covariance Sigma_BL must be SPD."""
    bl = _make_3asset_bl()
    P = np.array([[1.0, 0.0, -1.0]])
    Q = np.array([0.03])
    Omega = bl.default_omega(P)
    _, Sigma_bl = bl.posterior(P, Q, Omega)
    try:
        np.linalg.cholesky(Sigma_bl)
    except np.linalg.LinAlgError as exc:
        pytest.fail(f"Sigma_BL is not SPD: {exc}")


def test_view_confidence_sensitivity_dataframe_shape():
    """view_confidence_sensitivity() returns correctly shaped DataFrame."""
    bl = _make_3asset_bl()
    P = np.array([[1.0, -1.0, 0.0]])
    Q = np.array([0.02])
    scalings = [0.1, 0.5, 1.0, 2.0, 5.0]
    df = bl.view_confidence_sensitivity(P, Q, scalings)
    assert df.shape == (len(scalings), bl.n), (
        f"Expected ({len(scalings)}, {bl.n}), got {df.shape}"
    )
