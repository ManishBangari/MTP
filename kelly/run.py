"""Command-line interface for the Kelly Auction Simulation framework.

Usage:
    python -m kelly.run --target table1|fig3|fig4|fig5|fig6|fig7|all [--mode empirical|remark1|both]
"""

import argparse
import sys
from typing import Optional

from kelly.experiments import (
    run_table1,
    run_fig3,
    run_fig4,
    run_fig5,
    run_fig6,
    run_fig7,
    run_sensitivity,
    load_config,
)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run Kelly auction simulations and reproduce paper results."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["table1", "fig3", "fig4", "fig5", "fig6", "fig7", "sensitivity", "all"],
        help="Experiment or artifact to generate.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["empirical", "remark1", "both"],
        help="Step-size mode: 'empirical' (G=1), 'remark1' (G=a/eps+1), or 'both'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save output tables and figures.",
    )

    args = parser.parse_args(argv)
    config = load_config(args.config)

    modes = ["empirical", "remark1"] if args.mode == "both" else [args.mode]

    if args.target in ["table1", "all"]:
        for mode in modes:
            run_table1(config, output_dir=args.output_dir, g_bound_mode=mode)

    if args.target in ["fig3", "all"]:
        run_fig3(config, output_dir=args.output_dir, g_bound_mode="empirical")

    if args.target in ["fig4", "all"]:
        run_fig4(config, output_dir=args.output_dir, g_bound_mode="empirical")

    if args.target in ["fig5", "all"]:
        run_fig5(config, output_dir=args.output_dir, g_bound_mode="empirical")

    if args.target in ["fig6", "all"]:
        run_fig6(config, output_dir=args.output_dir, g_bound_mode="empirical")

    if args.target in ["fig7", "all"]:
        run_fig7(config, output_dir=args.output_dir, g_bound_mode="empirical")

    if args.target in ["sensitivity", "all"]:
        run_sensitivity(config, output_dir=args.output_dir, g_bound_mode="empirical")


if __name__ == "__main__":
    main()
