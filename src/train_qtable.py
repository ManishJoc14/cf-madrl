"""
Training loop for Q-Table multi-agent traffic signal control.

Runs one QTableAgent per junction, step-by-step through MultiAgentSumoEnv.
Saves logs to logs/qtable/training_logs.json and plots to plots/qtable/.
"""

import os
import warnings
import logging

from src.qtable_agent import QTableAgent
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from tools.utils import ensure_dir, log_metrics, Logger, scan_topology
from tools.visualize import plot_training


def train_qtable(config, args_rounds=None):
    """
    Train independent Q-Table agents in the shared SUMO environment.
    """

    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)

    Logger.header("Q-Table Multi-Agent Training")

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    model_dir = "saved_models/qtable"
    log_dir = "logs/qtable"
    plot_dir = "plots/qtable"
    log_file = os.path.join(log_dir, "training_logs.json")

    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(plot_dir)

    # ------------------------------------------------------------------
    # Training Parameters
    # ------------------------------------------------------------------
    local_steps = config["training"]["local_steps_per_round"]
    rounds = args_rounds if args_rounds else config["training"]["federated_rounds"]
    save_freq = config["training"].get("save_freq", 10)

    qcfg = config["rl"]["qtable"]

    # Validate required hyperparameters
    required_keys = [
        "lr",
        "gamma",
        "epsilon",
        "epsilon_min",
        "epsilon_decay",
        "n_bins",
        "bin_edges",
        "lr_decay",
    ]

    for key in required_keys:
        if key not in qcfg:
            raise ValueError(f"Missing required qtable config key: {key}")

    # ------------------------------------------------------------------
    # Phase 1: Environment Setup
    # ------------------------------------------------------------------
    Logger.section("Phase 1: Environment Discovery & Setup")

    max_lanes, max_phases = scan_topology(config)
    junction_ids = config["system"]["controlled_junctions"]

    Logger.narrative(
        f"Topology: {max_lanes} lanes, {max_phases} phases, {len(junction_ids)} junctions"
    )

    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
        }
    )

    # ------------------------------------------------------------------
    # Phase 2: Agent Initialization
    # ------------------------------------------------------------------
    obs, _ = env.reset()

    first_aid = junction_ids[0]
    n_actions = env.action_space[first_aid].n

    agents = {
        aid: QTableAgent(
            junction_id=aid,
            n_actions=n_actions,
            lr=qcfg["lr"],
            gamma=qcfg["gamma"],
            epsilon=qcfg["epsilon"],
            epsilon_min=qcfg["epsilon_min"],
            epsilon_decay=qcfg["epsilon_decay"],
            n_bins=qcfg["n_bins"],
            bin_edges=qcfg["bin_edges"],
        )
        for aid in junction_ids
    }

    # Load checkpoints if available
    for aid in junction_ids:
        ckpt = os.path.join(model_dir, f"{aid}.pkl")
        if os.path.exists(ckpt):
            agents[aid].load(ckpt)
            agents[aid].epsilon = qcfg["epsilon"]
            Logger.info(f"Loaded Q-table for {aid}")

    # Clear old logs
    if os.path.exists(log_file):
        os.remove(log_file)

    Logger.success(
        f"Ready | Agents: {len(junction_ids)}, Rounds: {rounds}, Steps/Round: {local_steps}"
    )

    # ------------------------------------------------------------------
    # Phase 3: Training Loop
    # ------------------------------------------------------------------
    for r in range(rounds):
        obs, _ = env.reset()
        done = False
        step = 0

        round_rewards = {aid: [] for aid in junction_ids}
        round_queues = {aid: [] for aid in junction_ids}
        round_waits = {aid: [] for aid in junction_ids}

        while step < local_steps and not done:
            actions = {aid: agents[aid].select_action(obs[aid]) for aid in junction_ids}

            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

            is_done = terminateds.get("__all__", False)

            for aid in junction_ids:
                agents[aid].update(
                    obs[aid],
                    actions[aid],
                    rewards[aid],
                    next_obs[aid],
                    is_done,
                )

                round_rewards[aid].append(rewards[aid])
                round_queues[aid].append(infos[aid].get("step_queue", 0.0))
                round_waits[aid].append(infos[aid].get("step_wait", 0.0))

            obs = next_obs
            done = is_done
            step += 1

        # -------------------------
        # Logging
        # -------------------------
        for aid in junction_ids:
            mean_reward = sum(round_rewards[aid]) / max(len(round_rewards[aid]), 1)
            mean_queue = sum(round_queues[aid]) / max(len(round_queues[aid]), 1)
            mean_wait = sum(round_waits[aid]) / max(len(round_waits[aid]), 1)

            log_metrics(
                {
                    "round": r + 1,
                    "agent": aid,
                    "mean_reward": float(mean_reward),
                    "mean_queue": float(mean_queue),
                    "mean_wait": float(mean_wait),
                    "epsilon": agents[aid].epsilon,
                    "status": "trained",
                },
                log_file,
            )

        # -------------------------
        # Decay Epsilon + LR
        # -------------------------
        for aid in junction_ids:
            agents[aid].epsilon = max(
                qcfg["epsilon_min"],
                agents[aid].epsilon * qcfg["epsilon_decay"],
            )

            agents[aid].lr = max(
                qcfg["lr"] * 0.001,
                agents[aid].lr * qcfg["lr_decay"],
            )

        avg_reward = sum(
            sum(round_rewards[aid]) / max(len(round_rewards[aid]), 1)
            for aid in junction_ids
        ) / len(junction_ids)

        Logger.success(
            f"Round {r + 1}/{rounds} | Avg Reward: {avg_reward:.2f} | ε: {agents[first_aid].epsilon:.4f}"
        )

        # Save periodically
        if (r + 1) % save_freq == 0:
            for aid in junction_ids:
                agents[aid].save(os.path.join(model_dir, f"{aid}.pkl"))
            Logger.info(f"Checkpoint saved at round {r + 1}")

    # Final save
    for aid in junction_ids:
        agents[aid].save(os.path.join(model_dir, f"{aid}.pkl"))

    Logger.success("Training complete. Models saved.")

    # Plot
    try:
        plot_training(log_file=log_file, output_dir=plot_dir, algo_name="Q-Learning")
    except Exception as e:
        Logger.warning(f"Plotting failed: {e}")

    env.close()
