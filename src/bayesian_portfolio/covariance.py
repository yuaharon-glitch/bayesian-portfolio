"""
Covariance estimators: Sample, Ledoit-Wolf OAS, Normal-Wishart conjugate.

References: Chen et al. (2010) IEEE Trans. Signal Processing 58(10); Murphy (2007).
"""

from __future__ import annotations

import numpy as np

from bayesian_portfolio.priors import NormalWishartPosterior, NormalWishartPrior


def _verify_spd(matrix: np.ndarray, name: str) -> None:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        min_eig = float(np.linalg.eigvalsh(matrix).min())
        raise RuntimeError(f"{name} not SPD: min eigenvalue = {min_eig:.4e}.")


def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    if np.any(vols <= 0):
        raise ValueError("Covariance has non-positive diagonal.")
    return cov / np.outer(vols, vols)


class SampleCovariance:
    r"""
    Biased MLE sample covariance: Sigma = (1/T) * Xc^T Xc.

    Divides by T, not T-1 — consistent with the scatter matrix convention
    used in the Normal-Wishart update and RMT Marchenko-Pastur parameterisation.

    Attributes
    ----------
    covariance_, correlation_, condition_number_ : fitted attributes
    n_observations_, n_assets_ : ints
    """

    def __init__(self) -> None:
        self.covariance_: np.ndarray | None = None
        self.correlation_: np.ndarray | None = None
        self.condition_number_: float | None = None
        self.n_observations_: int | None = None
        self.n_assets_: int | None = None

    def fit(self, returns: np.ndarray) -> SampleCovariance:
        """
        Parameters
        ----------
        returns : np.ndarray, shape (T, n)

        Raises
        ------
        ValueError  if T < n
        RuntimeError  if result is not SPD
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape
        if T < n:
            raise ValueError(f"T={T} < n={n}: sample covariance singular.")

        Xc  = returns - returns.mean(axis=0)
        cov = Xc.T @ Xc / T
        _verify_spd(cov, "SampleCovariance")

        eigs = np.linalg.eigvalsh(cov)
        self.covariance_     = cov
        self.correlation_    = _correlation_from_covariance(cov)
        self.condition_number_ = float(eigs.max() / eigs.min())
        self.n_observations_ = T
        self.n_assets_       = n
        return self


class LedoitWolfCovariance:
    r"""
    Oracle Approximating Shrinkage (OAS) covariance estimator.

    Shrinks toward mu*I:

    .. math::

        \hat{\Sigma} = (1 - \alpha^*) S + \alpha^* \mu^* I_n

    with the analytical shrinkage intensity (Chen et al. 2010):

    .. math::

        \alpha^* = \min\!\left(1,\;
        \frac{\tfrac{n+2}{T}\,\mathrm{tr}(S^2) + \mathrm{tr}(S)^2}
             {(T+n+2)\!\left(\mathrm{tr}(S^2) - \tfrac{\mathrm{tr}(S)^2}{n}\right)}
        \right)

    tr(S^2) computed as sum(S*S) — O(n^2) vs the O(n^3) matrix product.
    When S ∝ I the denominator vanishes; alpha is set to 0 in that case.

    Attributes
    ----------
    covariance_, correlation_, condition_number_ : fitted attributes
    shrinkage_intensity_ : float   alpha* in [0, 1]
    """

    def __init__(self) -> None:
        self.covariance_: np.ndarray | None = None
        self.correlation_: np.ndarray | None = None
        self.condition_number_: float | None = None
        self.shrinkage_intensity_: float | None = None
        self.n_observations_: int | None = None
        self.n_assets_: int | None = None

    @property
    def alpha_(self) -> float | None:
        return self.shrinkage_intensity_

    def fit(self, returns: np.ndarray) -> LedoitWolfCovariance:
        """
        Parameters
        ----------
        returns : np.ndarray, shape (T, n)

        Raises
        ------
        ValueError  if T < n
        RuntimeError  if result is not SPD
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape
        if T < n:
            raise ValueError(f"T={T} < n={n}: OAS requires T >= n.")

        Xc = returns - returns.mean(axis=0)
        S  = Xc.T @ Xc / T

        tr_S   = np.trace(S)
        tr_S2  = float(np.sum(S * S))   # sum(S_ij^2) = tr(S^2), O(n^2)
        tr_Ssq = tr_S ** 2

        num = (n + 2) / T * tr_S2 + tr_Ssq
        den = (T + n + 2) * (tr_S2 - tr_Ssq / n)

        # den = 0 when S ∝ I; any alpha gives the same result in that case
        alpha = 0.0 if abs(den) < 1e-15 else float(min(1.0, max(0.0, num / den)))

        mu_t = tr_S / n
        cov  = (1.0 - alpha) * S + alpha * mu_t * np.eye(n)
        _verify_spd(cov, "LedoitWolfCovariance")

        eigs = np.linalg.eigvalsh(cov)
        self.covariance_        = cov
        self.correlation_       = _correlation_from_covariance(cov)
        self.condition_number_  = float(eigs.max() / eigs.min())
        self.shrinkage_intensity_ = alpha
        self.n_observations_    = T
        self.n_assets_          = n
        return self


class NormalWishartCovariance:
    r"""
    Bayesian covariance via the Normal-Wishart posterior mean:

    .. math::

        \hat{\Sigma} = \Psi_n / (\nu_n - n - 1)

    Default prior: kappa_0=1, nu_0=n+2, Psi_0=I (weakly informative).

    Parameters
    ----------
    prior : NormalWishartPrior or None
        Pass None to use the default vague prior built at fit time.

    Attributes
    ----------
    covariance_, correlation_, condition_number_ : fitted attributes
    prior_strength_ratio_ : float   kappa_0 / kappa_n
    posterior_ : NormalWishartPosterior
    """

    def __init__(self, prior: NormalWishartPrior | None = None) -> None:
        self._user_prior = prior
        self.covariance_: np.ndarray | None = None
        self.correlation_: np.ndarray | None = None
        self.condition_number_: float | None = None
        self.prior_strength_ratio_: float | None = None
        self.posterior_: NormalWishartPosterior | None = None
        self.n_observations_: int | None = None
        self.n_assets_: int | None = None

    def fit(self, returns: np.ndarray) -> NormalWishartCovariance:
        """
        Parameters
        ----------
        returns : np.ndarray, shape (T, n)

        Raises
        ------
        ValueError  if T < n
        RuntimeError  if result is not SPD
        """
        returns = np.asarray(returns, dtype=float)
        T, n = returns.shape
        if T < n:
            raise ValueError(f"T={T} < n={n}: NW estimator requires T >= n.")

        prior = self._user_prior or NormalWishartPrior(
            mu_0=np.zeros(n),
            kappa_0=1.0,
            nu_0=float(n + 2),  # nu_0 = n+2 is the minimum for E[Sigma] to exist
            Psi_0=np.eye(n),
        )

        post = NormalWishartPosterior.update(prior, returns)
        cov  = post.posterior_covariance()
        _verify_spd(cov, "NormalWishartCovariance")

        eigs = np.linalg.eigvalsh(cov)
        self.covariance_          = cov
        self.correlation_         = _correlation_from_covariance(cov)
        self.condition_number_    = float(eigs.max() / eigs.min())
        self.prior_strength_ratio_ = post.prior_strength_ratio()
        self.posterior_           = post
        self.n_observations_      = T
        self.n_assets_            = n
        return self
