"""
Main entry point for CF-MADRL, Q-Table, and DQN training and evaluation.
"""

import argparse
import logging
import warnings

from tools.utils import load_config, Logger

# Global silencing
warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)


if __name__ == "__main__":
    """
    Entry point for training or evaluation of any agent type.

    Usage:
    python main.py --mode train                              # CF-MADRL (default)
    python main.py --mode train --rounds 50                 # CF-MADRL, 50 rounds
    python main.py --mode train --rounds 50 --agent qtable  # Q-Table
    python main.py --mode train --rounds 50 --agent dqn     # DQN
    python main.py --mode eval                              # Eval CF-MADRL
    python main.py --mode eval --agent qtable               # Eval Q-Table
    python main.py --mode eval --agent dqn                  # Eval DQN
    """

    # 1. Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Traffic Signal Control — Multi-Agent RL"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval"],
        help="Execution mode: 'train' or 'eval'",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of training rounds (overrides config). Train mode only.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="cfmadrl",
        choices=["cfmadrl", "qtable", "dqn"],
        help=(
            "Agent type to train/evaluate:\n"
            "  cfmadrl — CF-MADRL with RLlib PPO + Federated Learning (default)\n"
            "  qtable  — Independent Q-Table agents\n"
            "  dqn     — Independent DQN agents (PyTorch MLP + replay buffer)"
        ),
    )
    args = parser.parse_args()

    # 2. Load system configuration
    config = load_config("config.yaml")
    Logger.header(
        f"Traffic RL | Agent: {args.agent.upper()} | Mode: {args.mode.upper()}"
    )

    # 3. Route to the correct train/eval function
    if args.mode == "train":
        Logger.section(f"Launching {args.agent.upper()} Training")

        if args.agent == "cfmadrl":
            from src.train import train_rllib

            train_rllib(config, args_rounds=args.rounds)

        elif args.agent == "qtable":
            from src.train_qtable import train_qtable

            train_qtable(config, args_rounds=args.rounds)

        elif args.agent == "dqn":
            from src.train_dqn import train_dqn

            train_dqn(config, args_rounds=args.rounds)

    else:  # eval
        Logger.section(f"Launching {args.agent.upper()} Evaluation")

        if args.agent == "cfmadrl":
            from src.eval import evaluate_rl

            evaluate_rl(config)

        elif args.agent == "qtable":
            from src.eval_qtable import evaluate_qtable

            evaluate_qtable(config)

        elif args.agent == "dqn":
            from src.eval_dqn import evaluate_dqn

            evaluate_dqn(config)
