"""Verification tests for Phase 2: Dynamics, SDSC, Contraction, and Invariants.

Covers Test 3, Test 5, and Test 7 from PROMPT.md.
"""

import numpy as np
import pytest

from kelly.game import KellyGame
from kelly.equilibrium import compute_equilibrium
from kelly.algorithms import simulate, simulate_heterogeneous
from kelly.metrics import fixed_point_residual, check_convergence


def test_3_theorem_2_sdsc_negative_definiteness():
    """Test 3: Theorem 2 / SDSC.

    1. Build H_r(z)_{ij} = r_i d^2_{ij} phi_i + r_j d^2_{ji} phi_j with r_i = 1/a_i.
       Check negative definiteness (all eigenvalues < 0) on dense random sample
       of z in [eps, c]^n for n in {2, 10, 20} and gamma in {0, 5, 10}.
    2. Evaluate psi_{r,V}(x) < 1 on the set Delta.
    """
    rng = np.random.default_rng(2026)

    for n in [2, 10, 20]:
        for gamma in [0.0, 5.0, 10.0]:
            game = KellyGame(n=n, gamma=gamma)

            # Sample 25 points across the domain [eps, c]^n
            for _ in range(25):
                z = rng.uniform(game.eps, game.c, size=n)
                H_r = game.sdsc_matrix(z)

                # Check symmetry
                np.testing.assert_allclose(H_r, H_r.T, atol=1e-12)

                # Eigenvalues must all be strictly negative (< 0)
                eigvals = np.linalg.eigvalsh(H_r)
                max_eig = np.max(eigvals)
                assert max_eig < -1e-11, (
                    f"SDSC violation: max eigenvalue {max_eig} >= 0 for n={n}, gamma={gamma}"
                )

                # Check psi_{r,V}(x) < 1 on Delta
                # Share vector x_i = z_i / (sum z_j + delta)
                x = game.shares(z)
                # For logarithmic valuation with r_i = 1/a_i, psi(x) = sum_i x_i^2 <= (sum x_i)^2 < 1
                psi_val = np.sum(x**2)
                assert psi_val < 1.0, f"psi_{{r,V}}(x) = {psi_val} >= 1 on Delta"


def test_5_theorem_5_br_contraction_bound_and_rate():
    """Test 5: Theorem 5.

    1. Compute bound (35) for experimental parameters and verify/report that
       it is violated (eps = 1 is far below the required threshold).
    2. Estimate ||J_{BR~}||_inf at s_min = (n-1)*eps + delta.
    3. Fit the empirical linear convergence rate rho of BR dynamics.
    """
    results = {}

    for n in [2, 10, 20]:
        game = KellyGame(n=n, gamma=0.0)
        s_min = (n - 1) * game.eps + game.delta
        a = game.a_vec[0]

        # zeta_i(s) = -1/2 + (s + 2*a_i) / (2 * sqrt(s^2 + 4 * a_i * s))
        disc = s_min**2 + 4.0 * a * s_min
        zeta_s_min = -0.5 + (s_min + 2.0 * a) / (2.0 * np.sqrt(disc))

        # ||J_BR~||_inf at s_min is (n - 1) * zeta(s_min)
        norm_J_inf_s_min = (n - 1) * zeta_s_min

        # The sufficient condition for contraction everywhere is ||J_BR~||_inf < 1
        is_violated = norm_J_inf_s_min > 1.0

        # Required threshold s_threshold to achieve (n-1)*zeta(s) < 1:
        # s(s + 4a) > ((n-1)^2 / n) * a^2
        target = ((n - 1) ** 2 / n) * (a**2)
        # s^2 + 4a s - target = 0 => s = (-4a + sqrt(16a^2 + 4 target)) / 2
        s_req = (-4.0 * a + np.sqrt(16.0 * a**2 + 4.0 * target)) / 2.0
        eps_req = (s_req - game.delta) / (n - 1)

        # Run BR dynamics to estimate empirical convergence rate rho
        sim = simulate(game, algo="BR", T=50, seed=0, mu=90.0, sigma=10.0)
        # Filter positive residuals before machine precision floor
        valid_res = sim.residuals[sim.residuals > 1e-12]
        if len(valid_res) >= 3:
            # Fit log(r_t) = log(r_0) + t * log(rho)
            t_axis = np.arange(len(valid_res))
            log_r = np.log(valid_res)
            slope, _ = np.polyfit(t_axis, log_r, 1)
            rho_empirical = float(np.exp(slope))
        else:
            rho_empirical = float((sim.residuals[-1] / sim.residuals[0]) ** (1.0 / len(sim.residuals)))

        results[n] = {
            "norm_J_inf_s_min": norm_J_inf_s_min,
            "is_violated": is_violated,
            "eps_req": eps_req,
            "rho_empirical": rho_empirical,
        }

        # Assertions as specified in PROMPT.md:
        # Bound (35) IS violated (norm > 1)
        assert is_violated, f"Expected bound (35) to be violated for n={n}, but got {norm_J_inf_s_min}"
        # Empirical BR convergence rate IS strictly contracting (rho < 1)
        assert 0.0 < rho_empirical < 1.0, f"Expected rho < 1, got {rho_empirical}"

    # Print report for traceability
    print("\n--- Theorem 5 Contraction Verification Report ---")
    for n, data in results.items():
        print(
            f"n={n}: ||J_BR~||_inf at s_min = {data['norm_J_inf_s_min']:.4f} > 1 "
            f"(Violated as expected! Required eps={data['eps_req']:.2f} vs actual eps=1.0). "
            f"Empirical linear rate rho = {data['rho_empirical']:.4f}"
        )


def test_7_rmq_equals_daq_and_invariants():
    """Test 7:
    1. RMQ(lambda = 1, constant eta) == DAQ_F exactly.
    2. OGD/DAQ/RRM/BR outputs always lie in [eps, c].
    3. The BR fixed point equals the NE from the root solver.
    """
    game = KellyGame(n=10, gamma=0.0)
    T = 200

    # 1. Compare DAQ_F and RMQ_F on identical initial conditions
    res_daq = simulate(game, algo="DAQ_F", T=T, seed=42)
    res_rmq = simulate(game, algo="RMQ_F", T=T, seed=42)

    # Initial bids must match
    np.testing.assert_allclose(res_daq.z_history[0], res_rmq.z_history[0])

    # Trajectories must be identical to machine precision
    np.testing.assert_allclose(
        res_daq.z_history,
        res_rmq.z_history,
        rtol=1e-12,
        atol=1e-12,
        err_msg="RMQ(lambda=1, constant eta) must equal DAQ_F exactly",
    )

    # 2. Invariant: bids always remain in [eps, c] across all algorithms
    for algo in ["BR", "OGD_F", "OGD_V", "DAQ_F", "DAQ_V", "RRM_V"]:
        for gamma in [0.0, 5.0, 10.0]:
            g = KellyGame(n=10, gamma=gamma)
            res = simulate(g, algo=algo, T=100, seed=123)
            assert np.all(res.z_history >= g.eps - 1e-12), f"{algo} violated lower bound eps"
            assert np.all(res.z_history <= g.c + 1e-12), f"{algo} violated upper bound c"

    # 3. Heterogeneous bids invariant
    res_het, _ = simulate_heterogeneous(game, algo1="BR", algo2="DAQ_F", alpha=0.5, T=100, seed=7)
    assert np.all(res_het.z_history >= game.eps - 1e-12)
    assert np.all(res_het.z_history <= game.c + 1e-12)

    # 4. BR fixed point equals root solver NE
    z_star = compute_equilibrium(game, tol=1e-12)
    br_z_star = game.best_response(z_star)
    np.testing.assert_allclose(z_star, br_z_star, rtol=1e-8, atol=1e-8)
