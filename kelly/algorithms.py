"""Learning dynamics and simulation engine for Kelly Auction Games.

Implements synchronous updates for:
- Best Response (BR, eq. 19)
- Online Gradient Descent (OGD_F, OGD_V, eq. 14)
- Dual Averaging (DAQ_F, DAQ_V, eq. 15)
- Regularized Dual Averaging (RRM_V, eqs. 17-18)
- Heterogeneous mixtures (e.g., BR vs DAQ_F, BR vs OGD_F, DAQ_F vs OGD_F)
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np

from kelly.game import KellyGame
from kelly.metrics import fixed_point_residual, normalized_payoff, running_time_average, check_convergence


@dataclass
class SimulationResult:
    """Encapsulates the output of a game dynamics simulation."""
    z_history: np.ndarray          # Shape (T, n)
    residuals: np.ndarray          # Shape (T,)
    payoffs: np.ndarray            # Shape (T, n)
    norm_payoffs: np.ndarray       # Shape (T, n)
    avg_norm_payoffs: np.ndarray   # Shape (T, n)
    converged: bool
    iterations_to_conv: Union[int, float]
    algo_name: str


def initialize_bids(
    game: KellyGame,
    rng: np.random.Generator,
    mu: float = 90.0,
    sigma: float = 10.0,
) -> np.ndarray:
    """Draws initial bids z(1) ~ Normal(mu, sigma^2) clipped to [eps, c].

    Parameters
    ----------
    game : KellyGame
        Game instance.
    rng : np.random.Generator
        Numpy random generator.
    mu : float
        Mean of normal distribution (default 90.0).
    sigma : float
        Standard deviation of normal distribution (default 10.0).

    Returns
    -------
    z1 : np.ndarray
        Initial bid vector of shape (n,).
    """
    raw_bids = rng.normal(loc=mu, scale=sigma, size=game.n)
    return np.clip(raw_bids, game.eps, game.c)


def simulate(
    game: KellyGame,
    algo: str,
    T: int = 3000,
    seed: Optional[int] = 0,
    mu: float = 90.0,
    sigma: float = 10.0,
    z_init: Optional[np.ndarray] = None,
    conv_threshold: float = 1e-5,
) -> SimulationResult:
    """Runs a single simulation of algorithm 'algo' for T steps.

    Supported algorithms:
    - 'BR': Best Response dynamics
    - 'OGD_F': Agile Online Gradient Descent with fixed step-size
    - 'OGD_V': Agile Online Gradient Descent with variable step-size
    - 'DAQ_F': Lazy Dual Averaging with fixed step-size
    - 'DAQ_V': Lazy Dual Averaging with variable step-size
    - 'RRM_V': Regularized Dual Averaging with variable step-size
    - 'RMQ_F': RRM with fixed step-size and lambda=1 (should equal DAQ_F)
    """
    rng = np.random.default_rng(seed)

    if z_init is not None:
        z_curr = np.clip(np.array(z_init, dtype=np.float64), game.eps, game.c)
    else:
        z_curr = initialize_bids(game, rng=rng, mu=mu, sigma=sigma)

    # Pre-allocate trajectory storage
    z_history = np.zeros((T, game.n), dtype=np.float64)
    residuals = np.zeros(T, dtype=np.float64)

    # State variables for dual-averaging algorithms
    # Running sum of gradients for DAQ
    g_sum = np.zeros(game.n, dtype=np.float64)
    # Dual vector for RRM
    y_rrm = np.zeros(game.n, dtype=np.float64)

    # Fixed step size for _F variants
    eta_fixed = game.D_vec / (game.G_vec * np.sqrt(T))

    for t in range(1, T + 1):
        # Record current state at time t
        z_history[t - 1] = z_curr

        # Compute best response and residual r_t = ||BR(z(t)) - z(t)||_2
        br_curr = game.best_response(z_curr)
        res_t = float(np.linalg.norm(br_curr - z_curr))
        residuals[t - 1] = res_t

        if t == T:
            break

        # Compute gradient g_t = d_i phi_i(z(t))
        g_t = game.gradient(z_curr)

        # Update rule to compute z(t + 1)
        if algo == "BR":
            z_next = br_curr

        elif algo == "OGD_F":
            # z_i(t+1) = clip(z_i(t) + eta * g_t^i, eps, c)
            z_next = np.clip(z_curr + eta_fixed * g_t, game.eps, game.c)

        elif algo == "OGD_V":
            # eta_{t+1} = D_i / (G_i * sqrt(t+1)) or sqrt(t)
            eta_t = game.D_vec / (game.G_vec * np.sqrt(t))
            z_next = np.clip(z_curr + eta_t * g_t, game.eps, game.c)

        elif algo == "DAQ_F":
            # z_i(t+1) = clip(eta * sum_{s=1..t} g_s^i, eps, c)
            g_sum += g_t
            z_next = np.clip(eta_fixed * g_sum, game.eps, game.c)

        elif algo == "DAQ_V":
            # eta_{t+1} = D_i / (G_i * sqrt(t+1)) multiplying running sum
            g_sum += g_t
            eta_next = game.D_vec / (game.G_vec * np.sqrt(t + 1))
            z_next = np.clip(eta_next * g_sum, game.eps, game.c)

        elif algo == "RRM_V":
            # y_i(t) = y_i(t-1) + eta_t * g_t^i with eta_t = D_i / (G_i * sqrt(t))
            # z_i(t+1) = clip(lambda_i * y_i(t), eps, c), lambda_i = 1
            eta_t = game.D_vec / (game.G_vec * np.sqrt(t))
            y_rrm += eta_t * g_t
            z_next = np.clip(y_rrm, game.eps, game.c)

        elif algo == "RMQ_F":
            # Test variant: RRM with fixed step-size and lambda=1
            y_rrm += eta_fixed * g_t
            z_next = np.clip(y_rrm, game.eps, game.c)

        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        z_curr = z_next

    # Calculate metrics over the trajectory
    payoffs = game.payoffs(z_history)
    norm_payoffs = game.normalized_payoffs(z_history)
    avg_norm_payoffs = running_time_average(norm_payoffs, axis=0)

    converged, iter_conv = check_convergence(residuals, threshold=conv_threshold)

    return SimulationResult(
        z_history=z_history,
        residuals=residuals,
        payoffs=payoffs,
        norm_payoffs=norm_payoffs,
        avg_norm_payoffs=avg_norm_payoffs,
        converged=converged,
        iterations_to_conv=iter_conv,
        algo_name=algo,
    )


def simulate_heterogeneous(
    game: KellyGame,
    algo1: str,
    algo2: str,
    alpha: float,
    T: int = 3000,
    seed: Optional[int] = 0,
    mu: float = 90.0,
    sigma: float = 10.0,
    z_init: Optional[np.ndarray] = None,
    conv_threshold: float = 1e-5,
) -> Tuple[SimulationResult, int]:
    """Runs a heterogeneous simulation with fraction alpha of agents using algo1 and remainder algo2.

    Parameters
    ----------
    game : KellyGame
        Game instance.
    algo1 : str
        Algorithm for group 1 (lowest indices: 0 .. n1 - 1).
    algo2 : str
        Algorithm for group 2 (indices: n1 .. n - 1).
    alpha : float
        Fraction of agents in group 1, n1 = round(alpha * n).
    T : int
        Number of steps.
    seed : Optional[int]
        Random seed.
    mu : float
        Initial bid mean.
    sigma : float
        Initial bid std.

    Returns
    -------
    result : SimulationResult
        Trajectory and metrics result.
    n1 : int
        Number of agents in group 1.
    """
    n1 = int(np.round(alpha * game.n))
    rng = np.random.default_rng(seed)

    if z_init is not None:
        z_curr = np.clip(np.array(z_init, dtype=np.float64), game.eps, game.c)
    else:
        z_curr = initialize_bids(game, rng=rng, mu=mu, sigma=sigma)

    z_history = np.zeros((T, game.n), dtype=np.float64)
    residuals = np.zeros(T, dtype=np.float64)

    # State variables
    g_sum = np.zeros(game.n, dtype=np.float64)
    y_rrm = np.zeros(game.n, dtype=np.float64)
    eta_fixed = game.D_vec / (game.G_vec * np.sqrt(T))

    algos = [algo1] * n1 + [algo2] * (game.n - n1)

    for t in range(1, T + 1):
        z_history[t - 1] = z_curr

        br_curr = game.best_response(z_curr)
        residuals[t - 1] = float(np.linalg.norm(br_curr - z_curr))

        if t == T:
            break

        g_t = game.gradient(z_curr)
        z_next = np.zeros(game.n, dtype=np.float64)

        for i in range(game.n):
            a_name = algos[i]
            if a_name == "BR":
                z_next[i] = br_curr[i]
            elif a_name == "OGD_F":
                z_next[i] = np.clip(z_curr[i] + eta_fixed[i] * g_t[i], game.eps, game.c)
            elif a_name == "OGD_V":
                eta_t = game.D_vec[i] / (game.G_vec[i] * np.sqrt(t))
                z_next[i] = np.clip(z_curr[i] + eta_t * g_t[i], game.eps, game.c)
            elif a_name == "DAQ_F":
                g_sum[i] += g_t[i]
                z_next[i] = np.clip(eta_fixed[i] * g_sum[i], game.eps, game.c)
            elif a_name == "DAQ_V":
                g_sum[i] += g_t[i]
                eta_next = game.D_vec[i] / (game.G_vec[i] * np.sqrt(t + 1))
                z_next[i] = np.clip(eta_next * g_sum[i], game.eps, game.c)
            elif a_name == "RRM_V":
                eta_t = game.D_vec[i] / (game.G_vec[i] * np.sqrt(t))
                y_rrm[i] += eta_t * g_t[i]
                z_next[i] = np.clip(y_rrm[i], game.eps, game.c)
            else:
                raise ValueError(f"Unknown algorithm in heterogeneous mixture: {a_name}")

        z_curr = z_next

    payoffs = game.payoffs(z_history)
    norm_payoffs = game.normalized_payoffs(z_history)
    avg_norm_payoffs = running_time_average(norm_payoffs, axis=0)
    converged, iter_conv = check_convergence(residuals, threshold=conv_threshold)

    return (
        SimulationResult(
            z_history=z_history,
            residuals=residuals,
            payoffs=payoffs,
            norm_payoffs=norm_payoffs,
            avg_norm_payoffs=avg_norm_payoffs,
            converged=converged,
            iterations_to_conv=iter_conv,
            algo_name=f"{algo1}_vs_{algo2}_alpha{alpha}",
        ),
        n1,
    )
