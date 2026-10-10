"""Metrics and evaluation functions for Kelly auction dynamics.

Implements:
- Fixed-point residual r_t = ||BR(z(t)) - z(t)||_2 (Eq. 39)
- Normalized payoffs phi_bar_i(z)
- Running time-average normalized payoffs
- Variational Gap(z) = sum_i d_i phi_i(z) * (z*_i - z_i) / a_i (Eq. 28)
- Iterations to convergence check
"""

from typing import Tuple, Union
import numpy as np

from kelly.game import KellyGame


def fixed_point_residual(game: KellyGame, z: np.ndarray) -> Union[float, np.ndarray]:
    """Computes the fixed-point residual r_t = ||BR(z) - z||_2.

    Parameters
    ----------
    game : KellyGame
        Game instance.
    z : np.ndarray
        Bid vector of shape (n,) or trajectory of shape (T, n).

    Returns
    -------
    residual : float or np.ndarray
        Residual scalar or array of residuals for trajectory.
    """
    if z.ndim == 1:
        br = game.best_response(z)
        return float(np.linalg.norm(br - z))
    elif z.ndim == 2:
        # Trajectory along axis 0
        s = game.others_aggregate_s(z)
        br = game.best_response_from_s(s)
        return np.linalg.norm(br - z, axis=-1)
    else:
        raise ValueError(f"Unsupported bid array dimension: {z.ndim}")


def normalized_payoff(game: KellyGame, z: np.ndarray) -> np.ndarray:
    """Computes normalized payoff phi_bar_i(z) = (phi_i(z) - phi_min) / (phi_max - phi_min).

    Parameters
    ----------
    game : KellyGame
        Game instance.
    z : np.ndarray
        Bid vector of shape (n,) or trajectory of shape (..., n).

    Returns
    -------
    norm_phi : np.ndarray
        Normalized payoff array with same shape as z.
    """
    return game.normalized_payoffs(z)


def running_time_average(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Computes running time average: (1 / t) * sum_{s=1..t} data(s).

    Parameters
    ----------
    data : np.ndarray
        Trajectory data.
    axis : int
        Time axis (default 0).

    Returns
    -------
    time_avg : np.ndarray
        Running time-averaged array.
    """
    cum_sum = np.cumsum(data, axis=axis)
    t_steps = np.arange(1, data.shape[axis] + 1)
    # Reshape t_steps for broadcasting
    shape = [1] * data.ndim
    shape[axis] = -1
    t_grid = t_steps.reshape(shape)
    return cum_sum / t_grid


def gap(game: KellyGame, z: np.ndarray, z_star: np.ndarray) -> Union[float, np.ndarray]:
    """Computes the variational gap Gap(z) = sum_i d_i phi_i(z) * (z*_i - z_i) / a_i (Eq. 28).

    Parameters
    ----------
    game : KellyGame
        Game instance.
    z : np.ndarray
        Bid profile (n,) or trajectory (T, n).
    z_star : np.ndarray
        Nash Equilibrium bid profile (n,).

    Returns
    -------
    gap_val : float or np.ndarray
        Variational gap value.
    """
    g = game.gradient(z)
    diff = z_star - z
    weighted = g * diff / game.a_vec
    return np.sum(weighted, axis=-1)


def check_convergence(
    residuals: np.ndarray,
    threshold: float = 1e-5,
) -> Tuple[bool, Union[int, float]]:
    """Determines iterations to convergence: first t with r_t < threshold.

    Parameters
    ----------
    residuals : np.ndarray
        Array of residuals r_1, r_2, ..., r_T (1D array).
    threshold : float
        Tolerance (default 1e-5).

    Returns
    -------
    converged : bool
        True if r_t < threshold for some t.
    val : int or float
        Iteration t (1-indexed) if converged; otherwise r_T (the final residual).
    """
    conv_indices = np.where(residuals < threshold)[0]
    if len(conv_indices) > 0:
        # 1-indexed iteration
        return True, int(conv_indices[0] + 1)
    else:
        return False, float(residuals[-1])
