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


def _save_figure(fig, path_prefix: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f"{path_prefix}.pdf")
    png_path = os.path.join(save_dir, f"{path_prefix}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure {path_prefix} saved to {pdf_path} and {png_path}")


def plot_fig3(
    data: Dict[str, Dict[str, np.ndarray]],
    save_dir: str = "results/figures",
    t_max_gamma0_5: int = 500,
    t_max_gamma10: int = 600,
) -> None:
    """Generates Figure 3: Convergence speed (log-scale r_t vs t) for gamma in {0, 5, 10}."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    gammas = [0, 5, 10]
    subtitles = ["(a) $\\gamma = 0$", "(b) $\\gamma = 5$", "(c) $\\gamma = 10$"]
    t_maxs = [t_max_gamma0_5, t_max_gamma0_5, t_max_gamma10]
    legend_algos = ["DAQ_F", "DAQ_V", "OGD_F", "RRM_V", "OGD_V", "BR"]

    for idx, (gamma, subtitle, t_lim) in enumerate(zip(gammas, subtitles, t_maxs)):
        ax = axes[idx]
        gamma_data = data[str(gamma)]
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

        ax.axhline(1e-5, color="red", linestyle=":", linewidth=1.5, label=r"Threshold $10^{-5}$" if idx == 0 else "")
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1e2)
        ax.set_xlim(0, t_lim)
        ax.set_xlabel("Time step ($t$)", fontsize=11)
        if idx == 0:
            ax.set_ylabel(r"$\| \mathbf{BR}(\mathbf{z}(t)) - \mathbf{z}(t) \|_2$", fontsize=12)
        ax.set_title(subtitle, fontsize=12, pad=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    _save_figure(fig, "fig3", save_dir)


def plot_fig4(
    data: Dict[str, Dict[str, np.ndarray]],
    save_dir: str = "results/figures",
    t_max: int = 2500,
    ne_payoff: float = 0.655,
) -> None:
    """Generates Figure 4: time-average and instantaneous normalized payoff for the representative agent."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    legend_algos = ["DAQ_F", "DAQ_V", "OGD_F", "RRM_V", "OGD_V", "BR"]
    t_axis = np.arange(1, t_max + 1)

    for idx, key in enumerate(["avg", "inst"]):
        ax = axes[idx]
        for algo in legend_algos:
            if algo in data[key]:
                ser = data[key][algo][:t_max]
                style = ALGO_STYLES[algo]
                ax.plot(
                    t_axis,
                    ser,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markevery=150,
                    markersize=5,
                    linewidth=1.8,
                    alpha=0.9,
                )
        ax.axhline(ne_payoff, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax.set_xlim(0, t_max)
        ax.set_ylim(0.24, 0.78)
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx == 0:
            ax.set_ylabel("Time-average normalized payoff", fontsize=11)
            ax.set_title("(a) Time-average payoff", fontsize=12, pad=8)
        else:
            ax.set_ylabel("Instantaneous normalized payoff", fontsize=11)
            ax.set_title("(b) Instantaneous payoff", fontsize=12, pad=8)
        ax.set_xlabel("Time step ($t$)", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=7,
        frameon=True,
        framealpha=0.95,
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _save_figure(fig, "fig4", save_dir)


def plot_fig5(
    data: Dict[str, Dict[str, np.ndarray]],
    save_dir: str = "results/figures",
    ne_payoff: float = 0.655,
    ne_bid: float = 90.0,
) -> None:
    """Generates Figure 5: heterogeneous BR vs DAQ_F and BR vs OGD_F dynamics.

    Layout: 4 rows x 3 columns.
    Row 0: Instantaneous payoff for BR vs DAQ_F (alpha_BR in 10%, 80%, 90%)
    Row 1: Bid for BR vs DAQ_F (alpha_BR in 10%, 80%, 90%)
    Row 2: Instantaneous payoff for BR vs OGD_F (alpha_BR in 10%, 80%, 90%)
    Row 3: Bid for BR vs OGD_F (alpha_BR in 10%, 80%, 90%)
    Panels (a), (b), (d), (e) show t <= 200; panels (c), (f) show t <= 3000.
    """
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    alphas = [0.1, 0.8, 0.9]
    t_limits = [200, 200, 3000]
    mark_intervals = [10, 10, 150]

    # Block 1: BR vs DAQ_F
    titles_br_daq = [
        r"(a) $\mathrm{BR}$ vs $\mathrm{DAQ_F}$ ($\alpha_{\mathrm{BR}}=10\%$)",
        r"(b) $\mathrm{BR}$ vs $\mathrm{DAQ_F}$ ($\alpha_{\mathrm{BR}}=80\%$)",
        r"(c) $\mathrm{BR}$ vs $\mathrm{DAQ_F}$ ($\alpha_{\mathrm{BR}}=90\%$)",
    ]
    for col_idx, (alpha, t_lim, m_step, title) in enumerate(zip(alphas, t_limits, mark_intervals, titles_br_daq)):
        key = f"BR_DAQ_F_{alpha}"
        panel = data[key]
        t_axis = np.arange(1, t_lim + 1)
        p_br = panel["A1_payoff"][:t_lim]
        b_br = panel["A1_bid"][:t_lim]
        p_daq = panel["A2_payoff"][:t_lim]
        b_daq = panel["A2_bid"][:t_lim]

        # Top: Payoff (Row 0)
        ax_p = axes[0, col_idx]
        ax_p.plot(t_axis, p_br, color="#2ca02c", marker="*", markevery=m_step, markersize=6, linewidth=1.8, label=r"$\mathrm{BR}$")
        ax_p.plot(t_axis, p_daq, color="#1f77b4", marker="s", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{DAQ_F}$")
        ax_p.axhline(ne_payoff, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax_p.set_xlim(0, t_lim)
        ax_p.set_title(title, fontsize=11, pad=6)
        if col_idx == 0:
            ax_p.set_ylabel("Instantaneous payoff", fontsize=10)
        ax_p.grid(True, linestyle="--", alpha=0.35)

        # Bottom: Bid (Row 1)
        ax_b = axes[1, col_idx]
        ax_b.plot(t_axis, b_br, color="#2ca02c", marker="*", markevery=m_step, markersize=6, linewidth=1.8, label=r"$\mathrm{BR}$")
        ax_b.plot(t_axis, b_daq, color="#1f77b4", marker="s", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{DAQ_F}$")
        ax_b.axhline(ne_bid, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax_b.set_xlim(0, t_lim)
        if col_idx == 0:
            ax_b.set_ylabel("Bid ($z_i$)", fontsize=10)
        ax_b.set_xlabel("Time step ($t$)", fontsize=10)
        ax_b.grid(True, linestyle="--", alpha=0.35)

    # Block 2: BR vs OGD_F
    titles_br_ogd = [
        r"(d) $\mathrm{BR}$ vs $\mathrm{OGD_F}$ ($\alpha_{\mathrm{BR}}=10\%$)",
        r"(e) $\mathrm{BR}$ vs $\mathrm{OGD_F}$ ($\alpha_{\mathrm{BR}}=80\%$)",
        r"(f) $\mathrm{BR}$ vs $\mathrm{OGD_F}$ ($\alpha_{\mathrm{BR}}=90\%$)",
    ]
    for col_idx, (alpha, t_lim, m_step, title) in enumerate(zip(alphas, t_limits, mark_intervals, titles_br_ogd)):
        key = f"BR_OGD_F_{alpha}"
        panel = data[key]
        t_axis = np.arange(1, t_lim + 1)
        p_br = panel["A1_payoff"][:t_lim]
        b_br = panel["A1_bid"][:t_lim]
        p_ogd = panel["A2_payoff"][:t_lim]
        b_ogd = panel["A2_bid"][:t_lim]

        # Top: Payoff (Row 2)
        ax_p = axes[2, col_idx]
        ax_p.plot(t_axis, p_br, color="#2ca02c", marker="*", markevery=m_step, markersize=6, linewidth=1.8, label=r"$\mathrm{BR}$")
        ax_p.plot(t_axis, p_ogd, color="#ff7f0e", marker="^", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{OGD_F}$")
        ax_p.axhline(ne_payoff, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax_p.set_xlim(0, t_lim)
        ax_p.set_title(title, fontsize=11, pad=6)
        if col_idx == 0:
            ax_p.set_ylabel("Instantaneous payoff", fontsize=10)
        ax_p.grid(True, linestyle="--", alpha=0.35)

        # Bottom: Bid (Row 3)
        ax_b = axes[3, col_idx]
        ax_b.plot(t_axis, b_br, color="#2ca02c", marker="*", markevery=m_step, markersize=6, linewidth=1.8, label=r"$\mathrm{BR}$")
        ax_b.plot(t_axis, b_ogd, color="#ff7f0e", marker="^", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{OGD_F}$")
        ax_b.axhline(ne_bid, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax_b.set_xlim(0, t_lim)
        if col_idx == 0:
            ax_b.set_ylabel("Bid ($z_i$)", fontsize=10)
        ax_b.set_xlabel("Time step ($t$)", fontsize=10)
        ax_b.grid(True, linestyle="--", alpha=0.35)

    # Legends for each block
    handles_1, labels_1 = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles_1, labels_1, loc="upper right", fontsize=9, framealpha=0.9)
    handles_2, labels_2 = axes[2, 0].get_legend_handles_labels()
    axes[2, 0].legend(handles_2, labels_2, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save_figure(fig, "fig5", save_dir)


def plot_fig6(
    data: Dict[str, Dict[float, Dict[str, float]]],
    save_dir: str = "results/figures",
    prefix: str = "fig6",
    ne_payoff: float = 0.655,
) -> None:
    """Generates Figure 6: average payoff of representative agents against alpha.

    Subplots:
    (a) DAQ_F vs BR (alpha_BR in 10-80%)
    (b) OGD_F vs BR (alpha_BR in 10-80%)
    (c) DAQ_F vs OGD_F (alpha_OGD in 20-80%)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    panels_info = [
        {
            "key": "BR_DAQ_F",
            "title": r"(a) $\mathrm{DAQ_F}$ vs $\mathrm{BR}$",
            "xlabel": r"$\alpha_{\mathrm{BR}}$",
            "a1_name": r"$\mathrm{BR}$",
            "a1_color": "#2ca02c",
            "a1_marker": "*",
            "a2_name": r"$\mathrm{DAQ_F}$",
            "a2_color": "#1f77b4",
            "a2_marker": "s",
        },
        {
            "key": "BR_OGD_F",
            "title": r"(b) $\mathrm{OGD_F}$ vs $\mathrm{BR}$",
            "xlabel": r"$\alpha_{\mathrm{BR}}$",
            "a1_name": r"$\mathrm{BR}$",
            "a1_color": "#2ca02c",
            "a1_marker": "*",
            "a2_name": r"$\mathrm{OGD_F}$",
            "a2_color": "#ff7f0e",
            "a2_marker": "^",
        },
        {
            "key": "OGD_F_DAQ_F",
            "title": r"(c) $\mathrm{DAQ_F}$ vs $\mathrm{OGD_F}$",
            "xlabel": r"$\alpha_{\mathrm{OGD}}$",
            "a1_name": r"$\mathrm{OGD_F}$",
            "a1_color": "#ff7f0e",
            "a1_marker": "^",
            "a2_name": r"$\mathrm{DAQ_F}$",
            "a2_color": "#1f77b4",
            "a2_marker": "s",
        },
    ]

    for ax, p_info in zip(axes, panels_info):
        sub_dict = data[p_info["key"]]
        alphas = sorted(sub_dict.keys())
        vals_a1 = [sub_dict[a]["A1_avg_norm"] for a in alphas]
        vals_a2 = [sub_dict[a]["A2_avg_norm"] for a in alphas]

        ax.plot(
            alphas,
            vals_a1,
            label=p_info["a1_name"],
            color=p_info["a1_color"],
            marker=p_info["a1_marker"],
            markersize=7,
            linewidth=2.0,
        )
        ax.plot(
            alphas,
            vals_a2,
            label=p_info["a2_name"],
            color=p_info["a2_color"],
            marker=p_info["a2_marker"],
            markersize=6,
            linewidth=2.0,
            linestyle="--",
        )
        ax.axhline(ne_payoff, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax.set_title(p_info["title"], fontsize=12, pad=8)
        ax.set_xlabel(p_info["xlabel"], fontsize=11)
        ax.set_ylabel("Average normalized payoff", fontsize=11)
        ax.set_ylim(0.640, 0.658)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    _save_figure(fig, prefix, save_dir)


def plot_fig7(
    data: Dict[str, Dict[str, np.ndarray]],
    save_dir: str = "results/figures",
    ne_payoff: float = 0.655,
    ne_bid: float = 90.0,
) -> None:
    """Generates Figure 7: DAQ_F vs OGD_F across varying alpha mixtures.

    Layout: 2 rows x 5 columns.
    Top row: Bids (z_i) for alpha_DAQ in {10, 20, 50, 80, 90}%.
    Bottom row: Instantaneous payoff for alpha_DAQ in {10, 20, 50, 80, 90}%.
    Panels for 20/50/80% show t <= 200; panels for 10/90% show t <= 3000.
    """
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5))
    alphas = [0.1, 0.2, 0.5, 0.8, 0.9]
    t_limits = [3000, 200, 200, 200, 3000]
    mark_intervals = [150, 10, 10, 10, 150]
    titles = [
        r"(a) $\alpha_{\mathrm{DAQ}} = 10\%$",
        r"(b) $\alpha_{\mathrm{DAQ}} = 20\%$",
        r"(c) $\alpha_{\mathrm{DAQ}} = 50\%$",
        r"(d) $\alpha_{\mathrm{DAQ}} = 80\%$",
        r"(e) $\alpha_{\mathrm{DAQ}} = 90\%$",
    ]

    for col_idx, (alpha, t_lim, m_step, title) in enumerate(zip(alphas, t_limits, mark_intervals, titles)):
        info = data[f"alpha={alpha}"]
        t_axis = np.arange(1, t_lim + 1)
        b_daq = info["DAQ_F_bid"][:t_lim]
        p_daq = info["DAQ_F_payoff"][:t_lim]
        b_ogd = info["OGD_F_bid"][:t_lim]
        p_ogd = info["OGD_F_payoff"][:t_lim]

        # Top: Bid
        ax_b = axes[0, col_idx]
        ax_b.plot(t_axis, b_daq, color="#1f77b4", marker="s", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{DAQ_F}$")
        ax_b.plot(t_axis, b_ogd, color="#ff7f0e", marker="^", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{OGD_F}$")
        ax_b.axhline(ne_bid, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax_b.set_xlim(0, t_lim)
        ax_b.set_title(title, fontsize=11, pad=6)
        if col_idx == 0:
            ax_b.set_ylabel("Bid ($z_i$)", fontsize=11)
        ax_b.grid(True, linestyle="--", alpha=0.35)

        # Bottom: Instantaneous payoff
        ax_p = axes[1, col_idx]
        ax_p.plot(t_axis, p_daq, color="#1f77b4", marker="s", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{DAQ_F}$")
        ax_p.plot(t_axis, p_ogd, color="#ff7f0e", marker="^", markevery=m_step, markersize=5, linewidth=1.8, label=r"$\mathrm{OGD_F}$")
        ax_p.axhline(ne_payoff, color="red", linestyle="--", linewidth=1.5, label=r"$\mathrm{NE}$")
        ax_p.set_xlim(0, t_lim)
        if col_idx == 0:
            ax_p.set_ylabel("Instantaneous payoff", fontsize=11)
        ax_p.set_xlabel("Time step ($t$)", fontsize=11)
        ax_p.grid(True, linestyle="--", alpha=0.35)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=True,
        framealpha=0.95,
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_figure(fig, "fig7", save_dir)

