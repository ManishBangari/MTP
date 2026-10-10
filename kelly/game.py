"""Game model for Kelly Auction Game with logarithmic valuation.

Implements the proportional allocation auction game from:
"Learning in Proportional Allocation Auction Games" (Ben Mazziane et al., 2026).
"""

from typing import Tuple, Union, Optional
import numpy as np


class KellyGame:
    """Represents a Kelly auction game with logarithmic valuations.

    Parameters
    ----------
    n : int
        Number of players (default 10).
    a : float
        Base valuation parameter (default 100.0).
    gamma : float
        Heterogeneity slope (default 0.0).
    c : float
        Maximum bid (default 400.0).
    eps : float
        Minimum bid (default 1.0).
    delta : float
        Aggregate regularization constant (default 0.1).
    """

    def __init__(
        self,
        n: int = 10,
        a: float = 100.0,
        gamma: float = 0.0,
        c: float = 400.0,
        eps: float = 1.0,
        delta: float = 0.1,
        g_bound_mode: str = "empirical",
    ) -> None:
        self.n = int(n)
        self.a = float(a)
        self.gamma = float(gamma)
        self.c = float(c)
        self.eps = float(eps)
        self.delta = float(delta)
        self.g_bound_mode = g_bound_mode

        # Heterogeneity: a_i = max(a - i * gamma, 1) for i = 1..n
        # 1-indexed in mathematics: agent 1 has a - 1*gamma, etc.
        agent_indices = np.arange(1, self.n + 1, dtype=np.float64)
        self.a_vec = np.maximum(self.a - agent_indices * self.gamma, 1.0)

        # Bounds and Lipschitz constants:
        # D_i = c - eps
        self.D_vec = np.full(self.n, self.c - self.eps, dtype=np.float64)
        # G_i bound:
        # 'remark1' uses theoretical upper bound G_i = a_i / eps + 1 (Remark 1, eq. 16)
        # 'empirical' uses G_i = 1.0, matching the published numerical simulations
        if g_bound_mode == "remark1":
            self.G_vec = self.a_vec / self.eps + 1.0
        else:
            self.G_vec = np.ones(self.n, dtype=np.float64)

        # Cache analytical payoff bounds for normalization
        self.phi_min, self.phi_max = self._compute_payoff_bounds()

    def aggregate_S(self, z: np.ndarray) -> Union[float, np.ndarray]:
        """Aggregate sum S(z) = sum_j z_j + delta."""
        return np.sum(z, axis=-1, keepdims=False) + self.delta

    def others_aggregate_s(self, z: np.ndarray) -> np.ndarray:
        """Others' aggregate s_i(z) = sum_{j != i} z_j + delta = S(z) - z_i."""
        S = np.sum(z, axis=-1, keepdims=True) + self.delta
        return S - z

    def shares(self, z: np.ndarray) -> np.ndarray:
        """Resource share x_i(z) = z_i / S(z)."""
        S = np.sum(z, axis=-1, keepdims=True) + self.delta
        return z / S

    def payoffs(self, z: np.ndarray) -> np.ndarray:
        """Utility phi_i(z) = a_i ln(x_i(z)) - z_i."""
        S = np.sum(z, axis=-1, keepdims=True) + self.delta
        x = z / S
        return self.a_vec * np.log(x) - z

    def marginal_utility(self, z: np.ndarray) -> np.ndarray:
        """Gradient / Marginal utility d_i phi_i(z) = a_i (1/z_i - 1/S(z)) - 1."""
        S = np.sum(z, axis=-1, keepdims=True) + self.delta
        return self.a_vec * (1.0 / z - 1.0 / S) - 1.0

    def gradient(self, z: np.ndarray) -> np.ndarray:
        """Alias for marginal utility d_i phi_i(z)."""
        return self.marginal_utility(z)

    def best_response_from_s(self, s: np.ndarray) -> np.ndarray:
        """Best response BR_i(s) = clip((-s + sqrt(s^2 + 4 a_i s)) / 2, eps, c)."""
        # Eq. (20)
        disc = s**2 + 4.0 * self.a_vec * s
        br_unc = (-s + np.sqrt(np.maximum(disc, 0.0))) / 2.0
        return np.clip(br_unc, self.eps, self.c)

    def best_response(self, z: np.ndarray) -> np.ndarray:
        """Simultaneous best response BR(z) = (BR_i(s_i(z)))_i."""
        s = self.others_aggregate_s(z)
        return self.best_response_from_s(s)

    def jacobian_pseudogradient(self, z: np.ndarray) -> np.ndarray:
        """Jacobian matrix of pseudogradient g(z): J_ij = d^2_{ij} phi_i = d(g_i)/d(z_j).

        For i == j: d^2_{ii} phi_i = a_i * (-1/z_i^2 + 1/S(z)^2)
        For i != j: d^2_{ij} phi_i = a_i / S(z)^2
        """
        z = np.asarray(z, dtype=np.float64)
        S = np.sum(z) + self.delta
        # Outer product: a_i / S^2 for all (i, j)
        J = np.outer(self.a_vec, np.ones(self.n)) / (S**2)
        # Subtract a_i / z_i^2 on diagonal
        np.fill_diagonal(J, J.diagonal() - self.a_vec / (z**2))
        return J

    def sdsc_matrix(self, z: np.ndarray) -> np.ndarray:
        """Rosen's SDSC matrix H_r(z)_ij = r_i d^2_{ij} phi_i + r_j d^2_{ji} phi_j with r_i = 1/a_i.

        Because r_i * (a_i / S^2) = 1 / S^2 is symmetric across (i, j),
        H_r(z) = 2 * ( (1/S^2) * 1 1^T - diag(1 / z^2) ).
        """
        z = np.asarray(z, dtype=np.float64)
        S = np.sum(z) + self.delta
        H = (2.0 / (S**2)) * np.ones((self.n, self.n))
        diag_sub = 2.0 / (z**2)
        np.fill_diagonal(H, H.diagonal() - diag_sub)
        return H

    def _compute_payoff_bounds(self) -> Tuple[float, float]:
        """Analytically derives phi_min and phi_max over all players and joint actions.

        Max payoff: others bid eps, agent plays its best response.
        Min payoff: others bid c, agent evaluates corners {eps, c}.
        """
        # s_min: all others bid eps
        s_min = (self.n - 1) * self.eps + self.delta
        br_at_s_min = self.best_response_from_s(np.full(self.n, s_min))
        # Player i utility when playing BR against s_min
        S_max = br_at_s_min + s_min
        x_max = br_at_s_min / S_max
        phi_max_candidates = self.a_vec * np.log(x_max) - br_at_s_min
        phi_max = float(np.max(phi_max_candidates))

        # s_max: all others bid c
        s_max = (self.n - 1) * self.c + self.delta
        # At s_max, check corner bids eps and c for each player
        S_eps = self.eps + s_max
        x_eps = self.eps / S_eps
        phi_at_eps = self.a_vec * np.log(x_eps) - self.eps

        S_c = self.c + s_max
        x_c = self.c / S_c
        phi_at_c = self.a_vec * np.log(x_c) - self.c

        phi_min_candidates = np.minimum(phi_at_eps, phi_at_c)
        phi_min = float(np.min(phi_min_candidates))

        return phi_min, phi_max

    def normalized_payoffs(self, z: np.ndarray) -> np.ndarray:
        """Normalized payoff phi_bar_i(z) in [0, 1]."""
        phi = self.payoffs(z)
        denom = self.phi_max - self.phi_min
        if denom == 0.0:
            return np.zeros_like(phi)
        return (phi - self.phi_min) / denom
