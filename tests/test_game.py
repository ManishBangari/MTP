"""Verification tests for Phase 1: Game model and Equilibrium.

Covers Test 1, Test 2, Test 4, and Test 7 (NE part) from PROMPT.md.
"""

import numpy as np
import pytest
from scipy import optimize

from kelly.game import KellyGame
from kelly.equilibrium import (
    compute_ne_damped_br,
    compute_ne_root,
    compute_equilibrium,
)


def test_1_finite_difference_gradient_and_hessian():
    """Test 1: Finite-difference check of d_i phi_i and Hessian entries d^2_{ij} phi_i."""
    rng = np.random.default_rng(42)
    h = 1e-6

    for n in [2, 10, 20]:
        for gamma in [0.0, 5.0, 10.0]:
            game = KellyGame(n=n, gamma=gamma)
            # Sample random points strictly inside (eps, c)
            for _ in range(5):
                z = rng.uniform(game.eps + 10.0, game.c - 10.0, size=n)

                # 1. Check gradient d_i phi_i
                analytic_grad = game.gradient(z)
                num_grad = np.zeros(n)
                for i in range(n):
                    z_plus = z.copy()
                    z_minus = z.copy()
                    z_plus[i] += h
                    z_minus[i] -= h
                    phi_plus = game.payoffs(z_plus)[i]
                    phi_minus = game.payoffs(z_minus)[i]
                    num_grad[i] = (phi_plus - phi_minus) / (2.0 * h)

                np.testing.assert_allclose(
                    analytic_grad,
                    num_grad,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Gradient mismatch for n={n}, gamma={gamma}",
                )

                # 2. Check Hessian entries d^2_{ij} phi_i = d(g_i) / d(z_j)
                analytic_hess = game.jacobian_pseudogradient(z)
                num_hess = np.zeros((n, n))
                for j in range(n):
                    z_plus = z.copy()
                    z_minus = z.copy()
                    z_plus[j] += h
                    z_minus[j] -= h
                    g_plus = game.gradient(z_plus)
                    g_minus = game.gradient(z_minus)
                    num_hess[:, j] = (g_plus - g_minus) / (2.0 * h)

                np.testing.assert_allclose(
                    analytic_hess,
                    num_hess,
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"Hessian mismatch for n={n}, gamma={gamma}",
                )


def test_2_lemma_1_relations():
    """Test 2: Lemma 1: m(z)^2 d^2_{ij} U_i = f_i(x_i) if i=j, g_i(x_i) otherwise.

    For V_i(x) = a_i ln x:
    f_i(x) = (1-x)^2 V'' - 2(1-x) V' = a_i - a_i / x^2
    g_i(x) = -x(1-x) V'' + (2x-1) V' = a_i
    k_i(x) = g_i(x) - f_i(x) = a_i / x^2
    """
    rng = np.random.default_rng(123)

    for n in [2, 10, 20]:
        for gamma in [0.0, 5.0, 10.0]:
            game = KellyGame(n=n, gamma=gamma)
            for _ in range(5):
                z = rng.uniform(game.eps, game.c, size=n)
                m = game.aggregate_S(z)
                x = game.shares(z)
                J = game.jacobian_pseudogradient(z)

                # Scaled Hessian m(z)^2 * d^2_{ij} phi_i
                scaled_hess = (m**2) * J

                for i in range(n):
                    a_i = game.a_vec[i]
                    x_i = x[i]

                    # f_i(x_i) = a_i - a_i / x_i^2
                    f_i = a_i - a_i / (x_i**2)
                    # g_i(x_i) = a_i
                    g_i = a_i

                    # On diagonal (i == j):
                    np.testing.assert_allclose(
                        scaled_hess[i, i],
                        f_i,
                        rtol=1e-10,
                        err_msg="Lemma 1 failed on diagonal",
                    )

                    # Off diagonal (i != j):
                    for j in range(n):
                        if i != j:
                            np.testing.assert_allclose(
                                scaled_hess[i, j],
                                g_i,
                                rtol=1e-10,
                                err_msg="Lemma 1 failed off diagonal",
                            )

                    # Verify k_i = g_i - f_i = a_i / x_i^2
                    k_i = g_i - f_i
                    np.testing.assert_allclose(
                        k_i,
                        a_i / (x_i**2),
                        rtol=1e-10,
                        err_msg="Lemma 1 k_i relation failed",
                    )


def test_4_best_response_vs_numerical_argmax():
    """Test 4: BR closed form vs numerical argmax (scipy bounded scalar maximization) on random s."""
    rng = np.random.default_rng(999)

    for gamma in [0.0, 5.0, 10.0]:
        game = KellyGame(n=10, gamma=gamma)
        for _ in range(20):
            # Sample random others' aggregate s > 0
            # From very small to very large values
            s_val = rng.uniform(0.1, 4000.0)
            s_vec = np.full(game.n, s_val)

            br_analytic = game.best_response_from_s(s_vec)

            for i in range(game.n):
                a_i = game.a_vec[i]

                def neg_utility(z_i):
                    # phi_i(z_i, s) = a_i ln(z_i / (z_i + s)) - z_i
                    x_i = z_i / (z_i + s_val)
                    return -(a_i * np.log(x_i) - z_i)

                res = optimize.minimize_scalar(
                    neg_utility,
                    bounds=(game.eps, game.c),
                    method="bounded",
                    options={"xatol": 1e-10},
                )
                assert res.success
                np.testing.assert_allclose(
                    br_analytic[i],
                    res.x,
                    rtol=1e-6,
                    atol=1e-6,
                    err_msg=f"BR mismatch for agent {i}, s={s_val}",
                )


def test_7_ne_solvers_and_br_fixed_point():
    """Test 7 (NE part):
    - Damped BR fixed point equals scipy root solver
    - For gamma = 0, n = 10, expect z* ~ 90 per agent
    - BR dynamics converges to the unique NE
    - Normalized NE payoff for gamma = 0, n = 10 is ~0.655
    """
    # 1. Test symmetric case: n = 10, gamma = 0
    game_sym = KellyGame(n=10, gamma=0.0)
    z_star_damped, conv_damped, _ = compute_ne_damped_br(game_sym, tol=1e-12)
    z_star_root, conv_root = compute_ne_root(game_sym, tol=1e-12)
    z_star = compute_equilibrium(game_sym, tol=1e-12)

    assert conv_damped
    assert conv_root
    np.testing.assert_allclose(z_star_damped, z_star_root, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(z_star, z_star_root, rtol=1e-10, atol=1e-10)

    # For gamma = 0, n = 10, expect z* ~ 90
    assert np.all(np.abs(z_star - 90.0) < 0.1), f"Expected z* ~ 90, got {z_star}"

    # Normalized payoff at NE should be ~0.655
    norm_payoffs = game_sym.normalized_payoffs(z_star)
    for p in norm_payoffs:
        assert 0.65 <= p <= 0.66, f"Expected NE normalized payoff ~0.655, got {p}"

    # 2. Test asymmetric cases: gamma in {5, 10}, n in {2, 10, 20}
    for n in [2, 10, 20]:
        for gamma in [5.0, 10.0]:
            game = KellyGame(n=n, gamma=gamma)
            z_damped, conv_d, _ = compute_ne_damped_br(game, tol=1e-11)
            z_root, conv_r = compute_ne_root(game, tol=1e-11)

            assert conv_d
            assert conv_r
            np.testing.assert_allclose(
                z_damped,
                z_root,
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"NE mismatch for n={n}, gamma={gamma}",
            )

            # Check that z_star is indeed a fixed point of BR: ||BR(z*) - z*|| < 1e-6
            residual = np.linalg.norm(game.best_response(z_root) - z_root)
            assert residual < 1e-6, f"Residual at NE is {residual}"
