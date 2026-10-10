"""Experiment execution and artifact generation engine.

Produces:
- Table I (CSV, LaTeX, markdown comparison with paper values and relative errors)
- Figure 3 (homogeneous dynamics convergence trajectories across gamma in {0, 5, 10})
- Saves raw NPZ trajectories for rapid re-plotting without re-simulation.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from kelly.game import KellyGame
from kelly.algorithms import simulate, SimulationResult
from kelly.plotting import plot_fig3


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
