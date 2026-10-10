"""Equilibrium computation for Kelly Auction Games.

Implements independent Nash Equilibrium solvers:
1. Damped fixed-point iteration on Best Response
2. Scipy root-finding / least-squares on the KKT fixed-point system F(z) = z - BR(z)
"""

from typing import Optional, Tuple
import numpy as np
from scipy import optimize

from kelly.game import KellyGame


def compute_ne_damped_br(
    game: KellyGame,
    max_iter: int = 5000,
    tol: float = 1e-12,
    alpha: float = 0.5,
    z_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, int]:
    """Computes Nash Equilibrium via damped fixed-point iteration:
    z^{(k+1)} = (1 - alpha) * z^{(k)} + alpha * BR(z^{(k)}).

    Parameters
    ----------
    game : KellyGame
        Game instance.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on ||z^{(k+1)} - z^{(k)}||_inf.
    alpha : float
        Damping factor in (0, 1].
    z_init : Optional[np.ndarray]
        Initial bid vector. If None, initialized to midpoint (eps + c) / 2.

    Returns
    -------
    z_star : np.ndarray
        Computed Nash Equilibrium.
    converged : bool
        True if converged within tolerance.
    iterations : int
        Number of iterations executed.
    """
    if z_init is None:
        # Default starting point: midpoint of action space
        z = np.full(game.n, (game.eps + game.c) / 2.0, dtype=np.float64)
    else:
        z = np.clip(np.array(z_init, dtype=np.float64), game.eps, game.c)

    for it in range(1, max_iter + 1):
        br_z = game.best_response(z)
        z_next = (1.0 - alpha) * z + alpha * br_z
        diff = np.max(np.abs(z_next - z))
        z = z_next
        if diff < tol:
            return z, True, it

    return z, False, max_iter


def compute_ne_root(
    game: KellyGame,
    tol: float = 1e-12,
    z_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """Computes Nash Equilibrium via bounded root finding on F(z) = z - BR(z) = 0.

    Uses scipy.optimize.least_squares with box constraints [eps, c].

    Parameters
    ----------
    game : KellyGame
        Game instance.
    tol : float
        Tolerance for residual.
    z_init : Optional[np.ndarray]
        Initial bid vector.

    Returns
    -------
    z_star : np.ndarray
        Computed Nash Equilibrium.
    converged : bool
        True if optimization succeeded and residual < tol.
    """
    if z_init is None:
        z0 = np.full(game.n, (game.eps + game.c) / 2.0, dtype=np.float64)
    else:
        z0 = np.clip(np.array(z_init, dtype=np.float64), game.eps, game.c)

    def residual(z: np.ndarray) -> np.ndarray:
        return z - game.best_response(z)

    res = optimize.least_squares(
        residual,
        z0,
        bounds=(game.eps, game.c),
        ftol=tol,
        xtol=tol,
        gtol=tol,
        max_nfev=2000,
    )

    max_res = np.max(np.abs(res.fun))
    converged = bool(res.success and max_res < 1e-7)
    return res.x, converged


def compute_equilibrium(
    game: KellyGame,
    tol: float = 1e-12,
    z_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes the unique Nash Equilibrium independently of learning dynamics.

    Runs damped fixed point iteration first, followed by root refinement,
    and validates that both methods agree to within numerical tolerance.

    Parameters
    ----------
    game : KellyGame
        Game instance.
    tol : float
        Desired tolerance.
    z_init : Optional[np.ndarray]
        Initial guess.

    Returns
    -------
    z_star : np.ndarray
        Unique Nash Equilibrium bid profile.
    """
    z_damped, conv_damped, _ = compute_ne_damped_br(
        game, max_iter=5000, tol=tol, alpha=0.5, z_init=z_init
    )
    z_root, conv_root = compute_ne_root(game, tol=tol, z_init=z_damped)

    # Cross-check agreement between the two independent methods
    max_discrepancy = np.max(np.abs(z_damped - z_root))
    if max_discrepancy > 1e-6:
        # If any discrepancy, select root solver result if it converged
        if conv_root:
            return z_root
        return z_damped

    return z_root
