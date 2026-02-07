"""
Main entry point for CF-MADRL training and evaluation.
"""

import argparse
import logging
import warnings

from src.train import train_rllib
from src.eval import run_fixed_time_baseline, evaluate_rl, generate_comparison_plots
from tools.utils import load_config, Logger

# Global silencing
warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)


if __name__ == "__main__":
    """
    Entry point for CF-MADRL training or evaluation.

    Usage:
    python main.py --mode train        # Start training
    python main.py --mode eval         # Run evaluation only
    python main.py --mode train --rounds 50   # Train for 50 federated rounds
    """

    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser(description="CF-MADRL Main Script")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Execution mode: 'train' to run federated training, 'eval' to evaluate agents",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Optional: Number of federated training rounds (overrides config)",
    )
    args = parser.parse_args()

    # 2. Load system configuration
    config = load_config("config.yaml")
    Logger.header(f"CF-MADRL: Starting in '{args.mode}' mode")

    # 3. Execute chosen mode
    if args.mode == "train":
        Logger.section("Launching Federated Training")
        train_rllib(config, args_rounds=args.rounds)
    else:
        Logger.section("Launching Evaluation Pipeline")
        # Run fixed-time baseline
        fixed_metrics = run_fixed_time_baseline(config)
        
        # Run CF-MADRL evaluation
        madrl_metrics = evaluate_rl(config)
        
        # Compare and plot results
        generate_comparison_plots(fixed_metrics, madrl_metrics, config)
