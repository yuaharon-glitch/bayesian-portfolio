"""
Black-Litterman model: blend equilibrium returns with investor views.

Prior: mu ~ N(Pi, tau*Sigma), Pi = delta*Sigma*w_mkt.
Views: P*mu = Q + eps, eps ~ N(0, Omega).
Posterior: BL master formula via Gaussian-Gaussian conjugate update.

References: He & Litterman (1999); Idzorek (2005).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def implied_returns(
    Sigma: np.ndarray,
    w_mkt: np.ndarray,
    delta: float = 2.5,
) -> np.ndarray:
    r"""
    Equilibrium implied excess returns: Pi = delta * Sigma @ w_mkt.

    First-order condition of mean-variance optimisation at the market portfolio.
    Round-trip: z = solve(Sigma, Pi/delta); w = z/z.sum() recovers w_mkt.

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)
    w_mkt : np.ndarray, shape (n,)
    delta : float
        Risk aversion. Default 2.5.
    """
    return float(delta) * np.asarray(Sigma, dtype=float) @ np.asarray(w_mkt, dtype=float)


class BlackLitterman:
    r"""
    Black-Litterman posterior for returns given equilibrium + investor views.

    Parameters
    ----------
    Sigma : np.ndarray, shape (n, n)
    w_mkt : np.ndarray, shape (n,)
    delta : float
        Risk aversion. Default 2.5.
    tau : float
        Prior uncertainty scaling. Typical: 1/T or 0.05.

    Attributes
    ----------
    Pi : np.ndarray, shape (n,)
        Implied equilibrium returns = delta * Sigma @ w_mkt.
    """

    def __init__(
        self,
        Sigma: np.ndarray,
        w_mkt: np.ndarray,
        delta: float = 2.5,
        tau: float = 0.05,
    ) -> None:
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.w_mkt = np.asarray(w_mkt, dtype=float)
        self.delta = float(delta)
        self.tau   = float(tau)
        self.n     = len(w_mkt)
        self.Pi    = implied_returns(self.Sigma, self.w_mkt, self.delta)

    def posterior(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        BL posterior mean and covariance.

        .. math::

            M = (\tau\Sigma)^{-1} + P^\top \Omega^{-1} P

            \mu_{\text{BL}} = M^{-1}\bigl[(\tau\Sigma)^{-1}\Pi + P^\top\Omega^{-1}Q\bigr]

            \Sigma_{\text{BL}} = \Sigma + M^{-1}

        Parameters
        ----------
        P : np.ndarray, shape (k, n)
            Pick matrix. Pass np.zeros((0, n)) for no views.
        Q : np.ndarray, shape (k,)
        Omega : np.ndarray, shape (k, k)
            View uncertainty (diagonal SPD). Pass np.zeros((0, 0)) for no views.
        """
        P     = np.asarray(P, dtype=float)
        Q     = np.asarray(Q, dtype=float)
        Omega = np.asarray(Omega, dtype=float)
        k     = P.shape[0] if P.ndim == 2 else 0

        tau_S = self.tau * self.Sigma

        if k == 0:
            return self.Pi.copy(), self.Sigma + tau_S

        tS_inv_Pi   = np.linalg.solve(tau_S, self.Pi)
        Omega_inv_Q = np.linalg.solve(Omega, Q)
        rhs         = tS_inv_Pi + P.T @ Omega_inv_Q

        # Need (tau*Sigma)^{-1} as a matrix to assemble M — unavoidable full solve
        tS_inv      = np.linalg.solve(tau_S, np.eye(self.n))
        Omega_inv_P = np.linalg.solve(Omega, P)
        M           = tS_inv + P.T @ Omega_inv_P

        mu_BL    = np.linalg.solve(M, rhs)
        Sigma_BL = self.Sigma + np.linalg.solve(M, np.eye(self.n))

        return mu_BL, Sigma_BL

    def default_omega(self, P: np.ndarray) -> np.ndarray:
        r"""
        He-Litterman default: Omega = diag(P (tau*Sigma) P^T).

        Scales view uncertainty proportionally to the prior variance projected
        along each view direction.
        """
        P = np.asarray(P, dtype=float)
        return np.diag(np.diag(P @ (self.tau * self.Sigma) @ P.T))

    def view_confidence_sensitivity(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        omega_scalings: np.ndarray,
    ) -> pd.DataFrame:
        """
        Posterior means as Omega is scaled by each factor in omega_scalings.

        Scaling < 1 = higher confidence; > 1 = lower confidence.

        Returns
        -------
        pd.DataFrame, shape (len(omega_scalings), n)
        """
        P, Q = np.asarray(P, dtype=float), np.asarray(Q, dtype=float)
        Omega_base = self.default_omega(P)
        rows = {}
        for scale in omega_scalings:
            mu_bl, _ = self.posterior(P, Q, float(scale) * Omega_base)
            rows[float(scale)] = mu_bl
        return pd.DataFrame(rows, index=[f"asset_{i}" for i in range(self.n)]).T
