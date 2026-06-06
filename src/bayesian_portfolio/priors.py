"""
Normal-Wishart conjugate prior and posterior for multivariate return distributions.

Reference: Murphy (2007), Conjugate Bayesian analysis of the Gaussian distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RANDOM_SEED: int = 42


@dataclass
class NormalWishartPrior:
    r"""
    Parameters of a Normal-Wishart prior on (mu, Sigma).

    .. math::

        p(\mu, \Sigma) = \mathcal{N}(\mu \mid \mu_0, \Sigma / \kappa_0) \cdot
                         \mathcal{W}^{-1}(\Sigma \mid \nu_0, \Psi_0)

    Parameters
    ----------
    mu_0 : np.ndarray, shape (n,)
        Prior mean vector.
    kappa_0 : float
        Prior pseudo-count for the mean (> 0). Larger → stronger belief in mu_0.
    nu_0 : float
        Degrees of freedom for the Inverse-Wishart on Sigma. Must be > n-1.
        Use nu_0 >= n+1 so that E[Sigma] exists.
    Psi_0 : np.ndarray, shape (n, n)
        Scale matrix (SPD). E[Sigma] = Psi_0 / (nu_0 - n - 1) when nu_0 > n+1.
    """

    mu_0: np.ndarray
    kappa_0: float
    nu_0: float
    Psi_0: np.ndarray

    def __post_init__(self) -> None:
        self.mu_0 = np.asarray(self.mu_0, dtype=float)
        self.Psi_0 = np.asarray(self.Psi_0, dtype=float)
        n = len(self.mu_0)
        if self.Psi_0.shape != (n, n):
            raise ValueError(f"Psi_0 shape {self.Psi_0.shape} != ({n}, {n}).")
        if self.kappa_0 <= 0:
            raise ValueError(f"kappa_0 must be > 0, got {self.kappa_0}.")
        if self.nu_0 <= n - 1:
            raise ValueError(f"nu_0 must be > n-1={n-1}; got {self.nu_0}.")

    @property
    def n(self) -> int:
        """Dimension of the return vector."""
        return len(self.mu_0)

    def is_valid(self) -> bool:
        """True if nu_0 > n-1 and Psi_0 is SPD."""
        if self.nu_0 <= self.n - 1:
            return False
        try:
            np.linalg.cholesky(self.Psi_0)
        except np.linalg.LinAlgError:
            return False
        return True

    def prior_mean_of_mean(self) -> np.ndarray:
        """Return mu_0 (prior expected return vector)."""
        return self.mu_0.copy()

    def prior_mean_of_covariance(self) -> np.ndarray:
        r"""
        E[Sigma] = Psi_0 / (nu_0 - n - 1). Requires nu_0 > n+1.

        Raises
        ------
        ValueError
            If nu_0 <= n+1 (mean of Inverse-Wishart undefined).
        """
        n = self.n
        if self.nu_0 <= n + 1:
            raise ValueError(
                f"E[Sigma] requires nu_0 > n+1={n+1}; got {self.nu_0}."
            )
        return self.Psi_0 / (self.nu_0 - n - 1)


@dataclass
class NormalWishartPosterior:
    r"""
    Conjugate Normal-Wishart posterior after observing T returns.

    Closed-form update (Murphy 2007, eq. 250–254):

    .. math::

        \kappa_n = \kappa_0 + T, \quad \nu_n = \nu_0 + T

        \mu_n = \frac{\kappa_0 \mu_0 + T \bar{x}}{\kappa_n}

        \Psi_n = \Psi_0 + S + \frac{\kappa_0 T}{\kappa_n}
                 (\bar{x} - \mu_0)(\bar{x} - \mu_0)^\top

    where S is the raw scatter sum (NOT divided by T).
    """

    prior: NormalWishartPrior
    mu_n: np.ndarray
    kappa_n: float
    nu_n: float
    Psi_n: np.ndarray
    kappa_0: float
    n: int

    @classmethod
    def update(cls, prior: NormalWishartPrior, returns: np.ndarray) -> NormalWishartPosterior:
        r"""
        Compute the closed-form posterior given observed returns.

        Parameters
        ----------
        prior : NormalWishartPrior
        returns : np.ndarray, shape (T, n)

        Notes
        -----
        S = Xc.T @ Xc is the raw scatter (sum, not mean). Dividing by T
        here would break the conjugate update derivation.
        """
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2:
            raise ValueError(f"returns must be 2-D (T, n), got {returns.shape}.")
        T, n = returns.shape
        if T < 1:
            raise ValueError("Need at least 1 observation.")
        if n != prior.n:
            raise ValueError(f"returns has {n} assets but prior has dimension {prior.n}.")

        mu_0, kappa_0, nu_0, Psi_0 = prior.mu_0, prior.kappa_0, prior.nu_0, prior.Psi_0

        x_bar = returns.mean(axis=0)
        Xc = returns - x_bar
        S = Xc.T @ Xc   # scatter sum — not divided by T

        kappa_n = kappa_0 + T
        nu_n    = nu_0 + T
        mu_n    = (kappa_0 * mu_0 + T * x_bar) / kappa_n

        diff  = x_bar - mu_0
        Psi_n = Psi_0 + S + (kappa_0 * T / kappa_n) * np.outer(diff, diff)

        return cls(prior=prior, mu_n=mu_n, kappa_n=kappa_n, nu_n=nu_n,
                   Psi_n=Psi_n, kappa_0=kappa_0, n=n)

    def posterior_mean(self) -> np.ndarray:
        """E[mu | X] = mu_n."""
        return self.mu_n.copy()

    def posterior_covariance(self) -> np.ndarray:
        r"""
        E[Sigma | X] = Psi_n / (nu_n - n - 1). Requires nu_n > n+1.

        Raises
        ------
        ValueError
            If nu_n - n - 1 <= 0.
        """
        denom = self.nu_n - self.n - 1
        if denom <= 0:
            raise ValueError(
                f"nu_n - n - 1 = {denom} <= 0; increase nu_0 or collect more data."
            )
        return self.Psi_n / denom

    def predictive_mean(self) -> np.ndarray:
        """Predictive mean of x_{T+1} (equals mu_n)."""
        return self.mu_n.copy()

    def predictive_covariance(self) -> np.ndarray:
        r"""
        Predictive covariance of x_{T+1} under the marginal Student-t:

        .. math::

            \frac{\kappa_n + 1}{\kappa_n (\nu_n - n + 1)} \Psi_n

        Raises
        ------
        ValueError
            If nu_n - n + 1 <= 0.
        """
        denom = self.nu_n - self.n + 1
        if denom <= 0:
            raise ValueError(f"nu_n - n + 1 = {denom} <= 0.")
        return (self.kappa_n + 1) / (self.kappa_n * denom) * self.Psi_n

    def effective_sample_size(self) -> float:
        """kappa_n = kappa_0 + T (total pseudo-observations)."""
        return float(self.kappa_n)

    def prior_strength_ratio(self) -> float:
        """kappa_0 / kappa_n — prior fraction of posterior precision, in [0, 1)."""
        return float(self.kappa_0 / self.kappa_n)
