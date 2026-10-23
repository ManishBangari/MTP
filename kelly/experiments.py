"""Experiment execution and artifact generation engine.

Produces:
- Table I (CSV, LaTeX, markdown comparison with paper values and relative errors)
- Figure 3 (homogeneous dynamics convergence trajectories across gamma in {0, 5, 10})
- Figures 4-7 (payoff and heterogeneous dynamics plots)
- Saves raw NPZ trajectories for rapid re-plotting without re-simulation.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from kelly.game import KellyGame
from kelly.algorithms import simulate, simulate_heterogeneous, SimulationResult
from kelly.equilibrium import compute_equilibrium
from kelly.plotting import plot_fig3, plot_fig4, plot_fig5, plot_fig6, plot_fig7


PAPER_TABLE1_REF = {
    (0, 2):  {"BR": 15, "OGD_V": 37,   "OGD_F": 122, "DAQ_F": 195, "DAQ_V": 0.037, "RRM_V": 1682},
    (0, 10): {"BR": 7,  "OGD_V": 19,   "OGD_F": 240, "DAQ_F": 311, "DAQ_V": 1.05,  "RRM_V": 2291},
    (0, 20): {"BR": 6,  "OGD_V": 20,   "OGD_F": 252, "DAQ_F": 330, "DAQ_V": 1.73,  "RRM_V": 2361},
    (5, 2):  {"BR": 15, "OGD_V": 111,  "OGD_F": 127, "DAQ_F": 184, "DAQ_V": 0.10,  "RRM_V": 504},
    (5, 10): {"BR": 7,  "OGD_V": 40,   "OGD_F": 206, "DAQ_F": 274, "DAQ_V": 0.68,  "RRM_V": 2182},
    (5, 20): {"BR": 6,  "OGD_V": 1811, "OGD_F": 221, "DAQ_F": 289, "DAQ_V": 0.75,  "RRM_V": 2220},
    (10, 2): {"BR": 15, "OGD_V": 116,  "OGD_F": 117, "DAQ_F": 184, "DAQ_V": 0.097, "RRM_V": 498},
    (10, 10):{"BR": 8,  "OGD_V": 533,  "OGD_F": 185, "DAQ_F": 253, "DAQ_V": 0.45,  "RRM_V": 1604},
    (10, 20):{"BR": 8,  "OGD_V": 533,  "OGD_F": 194, "DAQ_F": 259, "DAQ_V": 0.45,  "RRM_V": 2062},
}


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Loads configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dataframe_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """Formats DataFrame to standard LaTeX tabular without external dependencies."""
    cols = " & ".join(df.columns)
    rows_str = []
    for _, row in df.iterrows():
        rows_str.append(" & ".join(str(val) for val in row.values) + " \\\\")
    body = "\n".join(rows_str)
    col_align = "c" * len(df.columns)
    return (
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{col_align}}}\n"
        f"\\hline\n"
        f"{cols} \\\\\n"
        f"\\hline\n"
        f"{body}\n"
        f"\\hline\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}\n"
    )


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Formats DataFrame to standard Markdown table without external dependencies."""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def format_table1_cell(val: float, is_daq_v: bool = False) -> str:
    """Formats cell value for Table I (integer or residual in parentheses)."""
    if is_daq_v:
        if val < 0.1:
            return f"({val:.2e})"
        else:
            return f"({val:.2f})"
    else:
        return str(int(round(val)))


def run_table1(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> pd.DataFrame:
    """Executes Table I experiment across n in {2, 10, 20} and gamma in {0, 5, 10}.

    Parameters
    ----------
    config : dict, optional
        Loaded configuration dictionary.
    output_dir : str
        Base results directory.
    g_bound_mode : str
        'empirical' (G=1.0, matching authors' numerical setup) or 'remark1' (theoretical G).

    Returns
    -------
    df_results : pd.DataFrame
        Formatted Table I results.
    """
    if config is None:
        config = load_config()

    table1_cfg = config.get("experiments", {}).get("table1", {})
    n_values = table1_cfg.get("n_values", [2, 10, 20])
    gamma_values = table1_cfg.get("gamma_values", [0, 5, 10])
    seeds = table1_cfg.get("seeds", list(range(10)))
    T = config.get("simulation", {}).get("T", 3000)
    mu = config.get("simulation", {}).get("mu", 90.0)
    sigma = config.get("simulation", {}).get("sigma", 10.0)
    conv_threshold = config.get("simulation", {}).get("conv_threshold", 1e-5)

    algos = ["BR", "OGD_V", "OGD_F", "DAQ_F", "DAQ_V", "RRM_V"]

    raw_data = {}
    rows = []
    comparison_rows = []

    print(f"=== Running Table I Simulation (g_bound_mode='{g_bound_mode}', {len(seeds)} runs/cell) ===")

    for gamma in gamma_values:
        for n in n_values:
            game = KellyGame(n=n, gamma=float(gamma), g_bound_mode=g_bound_mode)
            cell_metrics = {}

            row_entry = {"gamma": gamma, "n": n}
            comp_entry = {"gamma": gamma, "n": n}

            for algo in algos:
                algo_convs = []
                algo_residuals = []

                for s in seeds:
                    cache_key = f"g{gamma}_n{n}_{algo}_s{s}"
                    res = simulate(
                        game,
                        algo=algo,
                        T=T,
                        seed=s,
                        mu=mu,
                        sigma=sigma,
                        conv_threshold=conv_threshold,
                    )
                    raw_data[cache_key] = res.residuals
                    algo_convs.append(res.iterations_to_conv)
                    algo_residuals.append(res.residuals[-1])

                mean_val = float(np.mean(algo_convs))
                is_daq_v = (algo == "DAQ_V")
                cell_metrics[algo] = mean_val
                row_entry[algo] = format_table1_cell(mean_val, is_daq_v=is_daq_v)

                # Comparison with paper reference
                ref_val = PAPER_TABLE1_REF.get((gamma, n), {}).get(algo, None)
                if ref_val is not None:
                    if is_daq_v:
                        rel_err = abs(mean_val - ref_val) / ref_val * 100.0
                        comp_entry[f"{algo}_ours"] = f"({mean_val:.2f})"
                        comp_entry[f"{algo}_paper"] = f"({ref_val:.2f})"
                    else:
                        rel_err = abs(round(mean_val) - ref_val) / ref_val * 100.0
                        comp_entry[f"{algo}_ours"] = str(int(round(mean_val)))
                        comp_entry[f"{algo}_paper"] = str(ref_val)
                    comp_entry[f"{algo}_err%"] = f"{rel_err:.1f}%"

            rows.append(row_entry)
            comparison_rows.append(comp_entry)
            print(f"gamma={gamma:2d}, n={n:2d} -> " + " | ".join(f"{k}: {v}" for k, v in row_entry.items() if k not in ["gamma", "n"]))

    df_results = pd.DataFrame(rows)
    df_comp = pd.DataFrame(comparison_rows)

    # Save directories
    tables_dir = os.path.join(output_dir, "tables")
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    # 1. Save raw NPZ
    npz_path = os.path.join(raw_dir, f"table1_{g_bound_mode}.npz")
    np.savez_compressed(npz_path, **raw_data)
    print(f"Raw trajectories saved to {npz_path}")

    # 2. Save CSV
    csv_path = os.path.join(tables_dir, f"table1_{g_bound_mode}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Table I CSV saved to {csv_path}")

    # 3. Save LaTeX
    tex_path = os.path.join(tables_dir, f"table1_{g_bound_mode}.tex")
    latex_str = dataframe_to_latex(
        df_results,
        caption=f"Table I: Homogeneous dynamics convergence iterations ($r_t < 10^{{-5}}$, mode={g_bound_mode})",
        label=f"tab:table1_{g_bound_mode}",
    )
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"Table I LaTeX saved to {tex_path}")

    # 4. Save side-by-side Markdown comparison
    md_path = os.path.join(tables_dir, f"table1_comparison_{g_bound_mode}.md")
    with open(md_path, "w") as f:
        f.write(f"# Table I Comparison: Ours vs. Paper ({g_bound_mode})\n\n")
        f.write(dataframe_to_markdown(df_comp))
        f.write("\n")
    print(f"Comparison report saved to {md_path}")

    return df_results


def run_fig3(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> None:
    """Executes Figure 3 experiment: log-scale residual r_t vs t for gamma in {0, 5, 10} at n = 10.

    Parameters
    ----------
    config : dict, optional
        Loaded configuration dictionary.
    output_dir : str
        Base results directory.
    g_bound_mode : str
        'empirical' (G=1.0) or 'remark1'.
    """
    if config is None:
        config = load_config()

    fig3_cfg = config.get("experiments", {}).get("fig3", {})
    n = fig3_cfg.get("n", 10)
    gamma_values = fig3_cfg.get("gamma_values", [0, 5, 10])
    seeds = fig3_cfg.get("seeds", list(range(10)))
    T = config.get("simulation", {}).get("T", 3000)
    mu = config.get("simulation", {}).get("mu", 90.0)
    sigma = config.get("simulation", {}).get("sigma", 10.0)

    algos = ["DAQ_F", "DAQ_V", "OGD_F", "OGD_V", "RRM_V", "BR"]

    data_to_plot: Dict[str, Dict[str, np.ndarray]] = {}
    raw_arrays = {}

    print(f"\n=== Running Figure 3 Simulation (n={n}, gamma in {gamma_values}, {len(seeds)} runs/curve) ===")

    for gamma in gamma_values:
        game = KellyGame(n=n, gamma=float(gamma), g_bound_mode=g_bound_mode)
        gamma_key = str(gamma)
        data_to_plot[gamma_key] = {}

        for algo in algos:
            res_matrix = np.zeros((len(seeds), T), dtype=np.float64)
            for i, s in enumerate(seeds):
                sim = simulate(game, algo=algo, T=T, seed=s, mu=mu, sigma=sigma)
                res_matrix[i] = sim.residuals
                raw_arrays[f"fig3_g{gamma}_{algo}_s{s}"] = sim.residuals

            # Average over the independent runs
            mean_residuals = np.mean(res_matrix, axis=0)
            data_to_plot[gamma_key][algo] = mean_residuals

    # Save raw trajectories
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    npz_path = os.path.join(raw_dir, "fig3.npz")
    np.savez_compressed(npz_path, **raw_arrays)
    print(f"Figure 3 raw trajectory data saved to {npz_path}")

    # Generate plots
    fig_dir = os.path.join(output_dir, "figures")
    plot_fig3(data_to_plot, save_dir=fig_dir)


def run_fig4(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> None:
    """Generates Figure 4: time-average and instantaneous normalized payoff for a representative agent."""
    if config is None:
        config = load_config()

    fig4_cfg = config.get("experiments", {}).get("fig4", {})
    n = fig4_cfg.get("n", 10)
    gamma = float(fig4_cfg.get("gamma", 0.0))
    seeds = fig4_cfg.get("seeds", list(range(10)))
    T = config.get("simulation", {}).get("T", 3000)
    mu = config.get("simulation", {}).get("mu", 90.0)
    sigma = config.get("simulation", {}).get("sigma", 10.0)
    t_max = fig4_cfg.get("t_max", 2500)

    game = KellyGame(n=n, gamma=gamma, g_bound_mode=g_bound_mode)
    ne_bid = compute_equilibrium(game)
    ne_payoff = float(game.normalized_payoffs(ne_bid)[0])

    raw = {}
    data = {"avg": {}, "inst": {}}
    algos = ["DAQ_F", "DAQ_V", "OGD_F", "RRM_V", "OGD_V", "BR"]

    for algo in algos:
        res_matrix = np.zeros((len(seeds), T), dtype=np.float64)
        res_avg_matrix = np.zeros((len(seeds), T), dtype=np.float64)
        for i, seed in enumerate(seeds):
            sim = simulate(game, algo=algo, T=T, seed=seed, mu=mu, sigma=sigma)
            res_matrix[i] = sim.norm_payoffs[:, 0]
            res_avg_matrix[i] = sim.avg_norm_payoffs[:, 0]
            raw[f"fig4_{algo}_seed{seed}_avg"] = sim.avg_norm_payoffs[:, 0]
            raw[f"fig4_{algo}_seed{seed}_inst"] = sim.norm_payoffs[:, 0]

        data["avg"][algo] = np.mean(res_avg_matrix, axis=0)
        data["inst"][algo] = np.mean(res_matrix, axis=0)

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    np.savez_compressed(os.path.join(raw_dir, "fig4.npz"), **raw)

    fig_dir = os.path.join(output_dir, "figures")
    plot_fig4(data, save_dir=fig_dir, t_max=t_max, ne_payoff=ne_payoff)


def run_fig5(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> None:
    """Generates Figure 5: heterogeneous best-response vs DAQ/OGD mixtures."""
    if config is None:
        config = load_config()

    fig5_cfg = config.get("experiments", {}).get("fig5", {})
    n = fig5_cfg.get("n", 10)
    gamma = float(fig5_cfg.get("gamma", 0.0))
    seed = fig5_cfg.get("seed", 0)
    alphas = fig5_cfg.get("alphas", [0.1, 0.8, 0.9])
    T = config.get("simulation", {}).get("T", 3000)
    mu = config.get("simulation", {}).get("mu", 90.0)
    sigma = config.get("simulation", {}).get("sigma", 10.0)

    game = KellyGame(n=n, gamma=gamma, g_bound_mode=g_bound_mode)
    z_star = compute_equilibrium(game)
    ne_payoff = float(game.normalized_payoffs(z_star)[0])
    ne_bid = float(z_star[0])

    raw: Dict[str, np.ndarray] = {}
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for pair, algo1, algo2 in [("BR_DAQ_F", "BR", "DAQ_F"), ("BR_OGD_F", "BR", "OGD_F")]:
        for alpha in alphas:
            res, n1 = simulate_heterogeneous(
                game,
                algo1=algo1,
                algo2=algo2,
                alpha=float(alpha),
                T=T,
                seed=seed,
                mu=mu,
                sigma=sigma,
            )
            rep_a1 = 0
            rep_a2 = n1
            panel = {
                "A1_payoff": res.norm_payoffs[:, rep_a1],
                "A1_bid": res.z_history[:, rep_a1],
                "A2_payoff": res.norm_payoffs[:, rep_a2],
                "A2_bid": res.z_history[:, rep_a2],
            }
            data[f"{pair}_{alpha}"] = panel
            raw[f"{pair}_alpha{alpha}_A1_payoff"] = panel["A1_payoff"]
            raw[f"{pair}_alpha{alpha}_A1_bid"] = panel["A1_bid"]
            raw[f"{pair}_alpha{alpha}_A2_payoff"] = panel["A2_payoff"]
            raw[f"{pair}_alpha{alpha}_A2_bid"] = panel["A2_bid"]

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    np.savez_compressed(os.path.join(raw_dir, "fig5.npz"), **raw)

    # pass all data into plotting function for rendering
    plot_fig5(
        data,
        save_dir=os.path.join(output_dir, "figures"),
        ne_payoff=ne_payoff,
        ne_bid=ne_bid,
    )


def run_fig6(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> None:
    """Generates Figure 6: average payoff of representative agents against alpha.

    Generates both single seeded run (seed 0) and 10-run average (seeds 0-9) as
    requested in PROMPT.md.
    """
    if config is None:
        config = load_config()

    fig6_cfg = config.get("experiments", {}).get("fig6", {})
    n = fig6_cfg.get("n", 10)
    gamma = float(fig6_cfg.get("gamma", 0.0))
    seed_single = fig6_cfg.get("seed", 0)
    seeds_avg = fig6_cfg.get("seeds_avg", list(range(10)))
    T = config.get("simulation", {}).get("T", 3000)
    mu = config.get("simulation", {}).get("mu", 90.0)
    sigma = config.get("simulation", {}).get("sigma", 10.0)

    game = KellyGame(n=n, gamma=gamma, g_bound_mode=g_bound_mode)
    ne_bid = compute_equilibrium(game)
    ne_payoff = float(game.normalized_payoffs(ne_bid)[0])

    pairs_setup = [
        ("BR_DAQ_F", "BR", "DAQ_F", fig6_cfg.get("alphas_br_daq", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
        ("BR_OGD_F", "BR", "OGD_F", fig6_cfg.get("alphas_br_ogd", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
        ("OGD_F_DAQ_F", "OGD_F", "DAQ_F", fig6_cfg.get("alphas_ogd_daq", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])),
    ]

    summary_single: Dict[str, Dict[float, Dict[str, float]]] = {}
    summary_avg: Dict[str, Dict[float, Dict[str, float]]] = {}
    raw_single: Dict[str, np.ndarray] = {}
    raw_avg: Dict[str, np.ndarray] = {}

    print(f"\n=== Running Figure 6 Simulation (Single run + 10-run average) ===")

    for pair_key, algo1, algo2, alphas in pairs_setup:
        summary_single[pair_key] = {}
        summary_avg[pair_key] = {}

        for alpha in alphas:
            a_float = round(float(alpha), 2)

            # 1. Single seeded run (seed 0)
            res_single, n1_s = simulate_heterogeneous(
                game,
                algo1=algo1,
                algo2=algo2,
                alpha=a_float,
                T=T,
                seed=seed_single,
                mu=mu,
                sigma=sigma,
            )
            a1_idx = 0
            a2_idx = n1_s
            avg_a1_s = float(np.mean(res_single.norm_payoffs[:, a1_idx]))
            avg_a2_s = float(np.mean(res_single.norm_payoffs[:, a2_idx]))
            summary_single[pair_key][a_float] = {"A1_avg_norm": avg_a1_s, "A2_avg_norm": avg_a2_s}
            raw_single[f"{pair_key}_alpha{a_float}_A1"] = res_single.norm_payoffs[:, a1_idx]
            raw_single[f"{pair_key}_alpha{a_float}_A2"] = res_single.norm_payoffs[:, a2_idx]

            # 2. 10-run average (seeds 0-9)
            a1_runs = []
            a2_runs = []
            for s in seeds_avg:
                res_s, n1_r = simulate_heterogeneous(
                    game,
                    algo1=algo1,
                    algo2=algo2,
                    alpha=a_float,
                    T=T,
                    seed=s,
                    mu=mu,
                    sigma=sigma,
                )
                a1_runs.append(float(np.mean(res_s.norm_payoffs[:, 0])))
                a2_runs.append(float(np.mean(res_s.norm_payoffs[:, n1_r])))

            avg_a1_multi = float(np.mean(a1_runs))
            avg_a2_multi = float(np.mean(a2_runs))
            summary_avg[pair_key][a_float] = {"A1_avg_norm": avg_a1_multi, "A2_avg_norm": avg_a2_multi}
            raw_avg[f"{pair_key}_alpha{a_float}_A1_mean"] = np.array(a1_runs)
            raw_avg[f"{pair_key}_alpha{a_float}_A2_mean"] = np.array(a2_runs)

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    np.savez_compressed(os.path.join(raw_dir, "fig6_single.npz"), **raw_single)
    np.savez_compressed(os.path.join(raw_dir, "fig6_avg.npz"), **raw_avg)
    np.savez_compressed(os.path.join(raw_dir, "fig6.npz"), **raw_single)

    fig_dir = os.path.join(output_dir, "figures")
    plot_fig6(summary_single, save_dir=fig_dir, prefix="fig6_single", ne_payoff=ne_payoff)
    plot_fig6(summary_single, save_dir=fig_dir, prefix="fig6", ne_payoff=ne_payoff)
    plot_fig6(summary_avg, save_dir=fig_dir, prefix="fig6_avg", ne_payoff=ne_payoff)


def run_fig7(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> None:
    """Generates Figure 7: policy comparison between DAQ_F and OGD_F across heterogeneous fractions."""
    if config is None:
        config = load_config()

    fig7_cfg = config.get("experiments", {}).get("fig7", {})
    n = fig7_cfg.get("n", 10)
    gamma = float(fig7_cfg.get("gamma", 0.0))
    seed = fig7_cfg.get("seed", 0)
    alphas = fig7_cfg.get("alphas", [0.1, 0.2, 0.5, 0.8, 0.9])
    T = config.get("simulation", {}).get("T", 3000)
    mu = config.get("simulation", {}).get("mu", 90.0)
    sigma = config.get("simulation", {}).get("sigma", 10.0)

    game = KellyGame(n=n, gamma=gamma, g_bound_mode=g_bound_mode)
    z_star = compute_equilibrium(game)
    ne_payoff = float(game.normalized_payoffs(z_star)[0])
    ne_bid = float(z_star[0])

    raw: Dict[str, np.ndarray] = {}
    data: Dict[str, Dict[str, np.ndarray]] = {}

    for alpha in alphas:
        res, n1 = simulate_heterogeneous(
            game,
            algo1="DAQ_F",
            algo2="OGD_F",
            alpha=float(alpha),
            T=T,
            seed=seed,
            mu=mu,
            sigma=sigma,
        )
        data[f"alpha={alpha}"] = {
            "DAQ_F_payoff": res.norm_payoffs[:, 0],
            "DAQ_F_bid": res.z_history[:, 0],
            "OGD_F_payoff": res.norm_payoffs[:, n1],
            "OGD_F_bid": res.z_history[:, n1],
        }
        raw[f"alpha={alpha}_DAQ_F_payoff"] = res.norm_payoffs[:, 0]
        raw[f"alpha={alpha}_DAQ_F_bid"] = res.z_history[:, 0]
        raw[f"alpha={alpha}_OGD_F_payoff"] = res.norm_payoffs[:, n1]
        raw[f"alpha={alpha}_OGD_F_bid"] = res.z_history[:, n1]

    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    np.savez_compressed(os.path.join(raw_dir, "fig7.npz"), **raw)
    plot_fig7(
        data,
        save_dir=os.path.join(output_dir, "figures"),
        ne_payoff=ne_payoff,
        ne_bid=ne_bid,
    )


def run_sensitivity(
    config: Optional[dict] = None,
    output_dir: str = "results",
    g_bound_mode: str = "empirical",
) -> pd.DataFrame:
    """Runs the extra sensitivity analysis requested in the project prompt.

    This is intentionally labeled as 'not from the paper': re-run Table I for
    (mu, sigma) in {(90, 10), (90, 30), (200, 50)} and report the change from the
    baseline configuration.
    """
    if config is None:
        config = load_config()

    baseline_mu = config.get("simulation", {}).get("mu", 90.0)
    baseline_sigma = config.get("simulation", {}).get("sigma", 10.0)
    pairs = [(90.0, 10.0), (90.0, 30.0), (200.0, 50.0)]
    table_cfg = config.get("experiments", {}).get("table1", {})
    n_values = table_cfg.get("n_values", [2, 10, 20])
    gamma_values = table_cfg.get("gamma_values", [0, 5, 10])
    seeds = table_cfg.get("seeds", list(range(10)))
    conv_threshold = config.get("simulation", {}).get("conv_threshold", 1e-5)
    T = config.get("simulation", {}).get("T", 3000)

    rows = []
    for mu, sigma in pairs:
        table_rows = []
        for gamma in gamma_values:
            for n in n_values:
                game = KellyGame(n=n, gamma=float(gamma), g_bound_mode=g_bound_mode)
                data_for_row = {"mu": mu, "sigma": sigma, "gamma": gamma, "n": n}
                for algo in ["BR", "OGD_V", "OGD_F", "DAQ_F", "DAQ_V", "RRM_V"]:
                    vals = []
                    for s in seeds:
                        res = simulate(
                            game,
                            algo=algo,
                            T=T,
                            seed=s,
                            mu=mu,
                            sigma=sigma,
                            conv_threshold=conv_threshold,
                        )
                        vals.append(res.iterations_to_conv)
                    data_for_row[algo] = float(np.mean(vals))
                table_rows.append(data_for_row)
        rows.extend(table_rows)

    df = pd.DataFrame(rows)
    outdir = os.path.join(output_dir, "sensitivity")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"table1_sensitivity_{g_bound_mode}.csv")
    df.to_csv(csv_path, index=False)

    baseline = pd.DataFrame([
        {"mu": baseline_mu, "sigma": baseline_sigma, "gamma": gamma, "n": n}
        for gamma in gamma_values for n in n_values
    ])
    # The sensitivity result is reported in markdown alongside a brief note that it is not from the paper.
    markdown_lines = [
        "# Sensitivity check (not from the paper)",
        "",
        "This extra analysis re-runs Table I for the initial-bid settings (90, 10), (90, 30), and (200, 50).",
        "",
        "The baseline used in the paper is (mu, sigma) = (90, 10).",
        "",
        "| mu | sigma | gamma | n | BR | OGD_V | OGD_F | DAQ_F | DAQ_V | RRM_V |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df.iterrows():
        markdown_lines.append(
            "| " + " | ".join([str(row["mu"]), str(row["sigma"]), str(row["gamma"]), str(row["n"])] + [f"{row[c]:.2f}" for c in ["BR", "OGD_V", "OGD_F", "DAQ_F", "DAQ_V", "RRM_V"]]) + " |"
        )
    markdown_path = os.path.join(outdir, f"table1_sensitivity_{g_bound_mode}.md")
    with open(markdown_path, "w") as f:
        f.write("\n".join(markdown_lines) + "\n")

    return df
