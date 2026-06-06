"""Tests for bayesian_portfolio.covariance — three covariance estimators."""

from __future__ import annotations

import numpy as np
import pytest

from bayesian_portfolio.covariance import (
    LedoitWolfCovariance,
    NormalWishartCovariance,
    SampleCovariance,
)

RANDOM_SEED = 42

ALL_ESTIMATORS = [SampleCovariance, LedoitWolfCovariance, NormalWishartCovariance]


def test_lw_alpha_in_unit_interval():
    """OAS shrinkage intensity is always in [0, 1] for valid inputs."""
    rng = np.random.default_rng(RANDOM_SEED)
    test_cases = [
        (10, 100),    # low-dim, many obs
        (20, 200),    # moderate
        (5, 500),     # very tall
        (50, 300),    # wide-ish
        (30, 60),     # close to square
    ]
    for n, T in test_cases:
        lw = LedoitWolfCovariance().fit(rng.standard_normal((T, n)))
        alpha = lw.shrinkage_intensity_
        assert 0.0 <= alpha <= 1.0, f"alpha={alpha:.6f} out of [0, 1] for n={n}, T={T}"


def test_lw_alpha_near_one_high_dimensional():
    """OAS shrinkage intensity close to 1 when n≈T (near singular regime)."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    n, T = 40, 50
    lw = LedoitWolfCovariance().fit(rng.standard_normal((T, n)))
    assert 0.0 <= lw.shrinkage_intensity_ <= 1.0
    assert lw.shrinkage_intensity_ > 0.1, "Expected high shrinkage when T barely exceeds n"


def test_lw_alpha_near_zero_large_sample():
    """OAS shrinkage intensity close to 0 when T >> n (sample cov is reliable)."""
    rng = np.random.default_rng(RANDOM_SEED + 2)
    n, T = 5, 5000
    lw = LedoitWolfCovariance().fit(rng.standard_normal((T, n)))
    assert 0.0 <= lw.shrinkage_intensity_ <= 1.0


def test_raises_when_T_lt_n():
    """SampleCovariance, LedoitWolfCovariance, NormalWishartCovariance all
    raise ValueError when T < n (under-determined)."""
    rng = np.random.default_rng(RANDOM_SEED)
    bad = rng.standard_normal((50, 100))      # T=50 < n=100
    for cls in ALL_ESTIMATORS:
        with pytest.raises(ValueError, match=r"T=50 < n=100"):
            cls().fit(bad)


def test_raises_exact_T_equals_n_minus_one():
    """Raises when T = n-1 (one too few)."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = 20
    bad = rng.standard_normal((n - 1, n))
    for cls in ALL_ESTIMATORS:
        with pytest.raises(ValueError):
            cls().fit(bad)


def _fit_all(T: int = 200, n: int = 10, seed: int = RANDOM_SEED):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, n))
    return {
        "sample": SampleCovariance().fit(X),
        "lw": LedoitWolfCovariance().fit(X),
        "nw": NormalWishartCovariance().fit(X),
    }


def test_covariance_spd():
    """All estimators produce SPD covariance (Cholesky succeeds)."""
    fitted = _fit_all()
    for name, est in fitted.items():
        try:
            np.linalg.cholesky(est.covariance_)
        except np.linalg.LinAlgError:
            pytest.fail(f"{name} covariance_ is not SPD")


def test_correlation_diagonal_ones():
    """All correlation matrices have diagonal = 1."""
    fitted = _fit_all()
    for name, est in fitted.items():
        diag = np.diag(est.correlation_)
        assert np.allclose(diag, 1.0, atol=1e-12), (
            f"{name} correlation diagonal not 1: {diag}"
        )


def test_condition_number_positive():
    """All estimators store a positive condition number."""
    fitted = _fit_all()
    for name, est in fitted.items():
        assert est.condition_number_ > 0, f"{name} condition_number_ not positive"


def test_lw_better_conditioned_than_sample():
    """LW covariance should have lower condition number than sample covariance
    in the high-dimensional near-singular regime."""
    rng = np.random.default_rng(RANDOM_SEED)
    n, T = 30, 40    # close to singular
    X = rng.standard_normal((T, n))
    sc = SampleCovariance().fit(X)
    lw = LedoitWolfCovariance().fit(X)
    assert lw.condition_number_ < sc.condition_number_, (
        f"LW ({lw.condition_number_:.1f}) not better conditioned than "
        f"Sample ({sc.condition_number_:.1f})"
    )


def test_nw_prior_strength_ratio_in_unit_interval():
    """NW prior_strength_ratio_ ∈ (0, 1)."""
    rng = np.random.default_rng(RANDOM_SEED)
    nw = NormalWishartCovariance().fit(rng.standard_normal((100, 10)))
    assert 0.0 < nw.prior_strength_ratio_ < 1.0
