"""Plotting module for reproducing figures from Ben Mazziane et al. (2026).

Follows styling guidelines:
- Paper legend order and naming
- Markers: square = DAQ, triangle = OGD, star = BR, diamond = RRM
- High-contrast colors
- PDF and PNG output at 300 dpi
- Optional cached NPZ reloading
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


# Style definitions matching the paper
ALGO_STYLES = {
    "DAQ_F": {"label": r"$\mathrm{DAQ_F}$", "color": "#1f77b4", "marker": "s", "linestyle": "-", "markevery": 30},
    "DAQ_V": {"label": r"$\mathrm{DAQ_V}$", "color": "#17becf", "marker": "s", "linestyle": "--", "markevery": 30},
    "RRM_V": {"label": r"$\mathrm{RRM_V}$", "color": "#9467bd", "marker": "d", "linestyle": "-", "markevery": 30},
    "OGD_F": {"label": r"$\mathrm{OGD_F}$", "color": "#ff7f0e", "marker": "^", "linestyle": "-", "markevery": 30},
    "OGD_V": {"label": r"$\mathrm{OGD_V}$", "color": "#d62728", "marker": "v", "linestyle": "--", "markevery": 30},
    "BR":    {"label": r"$\mathrm{BR}$",    "color": "#2ca02c", "marker": "*", "linestyle": "-", "markevery": 10},
}


def plot_fig3(
    data: Dict[str, Dict[str, np.ndarray]],
    save_dir: str = "results/figures",
    t_max_gamma0_5: int = 500,
    t_max_gamma10: int = 600,
) -> None:
    """Generates Figure 3: Convergence speed (log-scale r_t vs t) for gamma in {0, 5, 10}.

    Parameters
    ----------
    data : dict
        Nested dictionary data[gamma_str][algo] = array of shape (T,) of averaged residuals.
    save_dir : str
        Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    gammas = [0, 5, 10]
    subtitles = ["(a) $\\gamma = 0$", "(b) $\\gamma = 5$", "(c) $\\gamma = 10$"]
    t_maxs = [t_max_gamma0_5, t_max_gamma0_5, t_max_gamma10]

    # Legend order matching paper
    legend_algos = ["DAQ_F", "DAQ_V", "OGD_F", "RRM_V", "OGD_V", "BR"]

    for idx, (gamma, subtitle, t_lim) in enumerate(zip(gammas, subtitles, t_maxs)):
        ax = axes[idx]
        gamma_key = str(gamma)
        gamma_data = data[gamma_key]

        t_axis = np.arange(1, t_lim + 1)

        for algo in legend_algos:
            if algo in gamma_data:
                res_series = gamma_data[algo][:t_lim]
                style = ALGO_STYLES.get(algo, {})
                ax.plot(
                    t_axis,
                    res_series,
                    label=style.get("label", algo),
                    color=style.get("color"),
                    marker=style.get("marker"),
                    linestyle=style.get("linestyle", "-"),
                    markevery=style.get("markevery", 25),
                    markersize=6,
                    linewidth=1.8,
                    alpha=0.9,
                )

        # Convergence threshold line at 1e-5
        ax.axhline(1e-5, color="red", linestyle=":", linewidth=1.5, label=r"Threshold $10^{-5}$" if idx == 0 else "")

        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e2)
        ax.set_xlim(0, t_lim)
        ax.set_xlabel("Time step ($t$)", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"$\| \mathbf{BR}(\mathbf{z}(t)) - \mathbf{z}(t) \|_2$", fontsize=12)
        ax.set_title(subtitle, fontsize=12, pad=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Place legend above or on the first plot
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    pdf_path = os.path.join(save_dir, "fig3.pdf")
    png_path = os.path.join(save_dir, "fig3.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3 saved to {pdf_path} and {png_path}")
