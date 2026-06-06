"""Bayesian portfolio optimization: Black-Litterman, Normal-Wishart, closed-form KKT."""

from __future__ import annotations

from bayesian_portfolio.black_litterman import BlackLitterman, implied_returns
from bayesian_portfolio.covariance import (
    LedoitWolfCovariance,
    NormalWishartCovariance,
    SampleCovariance,
)
from bayesian_portfolio.evaluation import WalkForwardEvaluator, diebold_mariano_test
from bayesian_portfolio.frontier import (
    EfficientFrontier,
    FrontierComparison,
    max_sharpe_weights,
    min_variance_weights,
    target_return_weights,
)
from bayesian_portfolio.priors import NormalWishartPosterior, NormalWishartPrior

__version__ = "0.1.0"

__all__ = [
    "NormalWishartPrior",
    "NormalWishartPosterior",
    "BlackLitterman",
    "implied_returns",
    "SampleCovariance",
    "LedoitWolfCovariance",
    "NormalWishartCovariance",
    "EfficientFrontier",
    "FrontierComparison",
    "min_variance_weights",
    "max_sharpe_weights",
    "target_return_weights",
    "WalkForwardEvaluator",
    "diebold_mariano_test",
]
