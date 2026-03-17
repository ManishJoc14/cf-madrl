"""
Training entry for CF-MADRL using RLlib multi-agent PPO.
"""

import os
import logging
import warnings
import torch
import numpy as np

from src.agent_manager import AgentManager
from src.federation import ClusteredFederatedServer
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from tools.visualize import plot_training, plot_clusters
from tools.utils import ensure_dir, log_metrics, Logger, scan_topology


def train_rllib(config, args_rounds=None):
    """
    Clustered Federated Multi-Agent RL training using RLlib PPO.

    Each agent controls a junction in SUMO. Agents are trained locally
    for a number of steps per round, then their weights are clustered
    and aggregated across clusters. Training uses patience to allow
    early stopping if convergence is detected.
    """

    # ---- Silence warnings and logging ----
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)

    Logger.header("CF-MADRL Federated Training")

    # ---- Prepare directories for models and logs ----
    model_dir = os.path.abspath(config["system"]["model_save_path"])
    base_log_dir = config["system"].get("log_dir", "logs")
    log_dir = os.path.join(base_log_dir, "cfmadrl")
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    # ---- Load training hyperparameters ----
    local_steps = config["training"]["local_steps_per_round"]
    rounds = args_rounds if args_rounds else config["training"]["federated_rounds"]
    save_freq = config["training"].get("save_freq", 5)
    n_clusters = config["training"]["n_clusters"]

    # ---- Load patience configuration for early stopping ----
    patience_cfg = config["training"].get("patience", {})
    patience_enabled = patience_cfg.get("enabled", False)
    patience_epochs = patience_cfg.get("epochs", 10)
    min_reward_delta = patience_cfg.get("min_reward_delta", 0.05)
    patience_counter = 0
    best_reward = -np.inf  # Tracks best observed average reward

    # ---- Device information ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Logger.info(f"Training Device: {device.upper()}")
    if device == "cuda":
        Logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Discover environment topology ----
    max_lanes, max_phases = scan_topology(config)
    junction_ids = config["system"]["controlled_junctions"]

    # ---- Initialize SUMO multi-agent environment ----
    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
        }
    )

    # ---- Initialize RLlib agent manager ----
    agent_manager = AgentManager(env, config, junction_ids)

    # ---- Check for existing checkpoints and resume if available ----
    is_resuming = os.path.exists(model_dir) and any(os.scandir(model_dir))
    if is_resuming:
        try:
            agent_manager.load(model_dir)
            Logger.success("Checkpoint loaded.")
        except Exception as e:
            Logger.warning(f"Checkpoint load failed: {e}")
            agent_manager.build()
    else:
        agent_manager.build()

    # ---- Initialize federated server for clustering agents ----
    server = ClusteredFederatedServer(n_clusters=n_clusters)

    # ---- Prepare log file ----
    log_file = os.path.join(log_dir, "training_logs.json")
    if not is_resuming and os.path.exists(log_file):
        os.remove(log_file)

    Logger.success(
        f"System Ready | Agents: {len(junction_ids)} | Clusters: {n_clusters} | Rounds: {rounds}"
    )

    # ---- Main training loop ----
    for r in range(rounds):
        Logger.round_banner(r + 1, rounds)

        # Compute number of training iterations for this round
        batch_size = config["rl"].get("train_batch_size", 512)
        num_iterations = max(1, local_steps // batch_size)

        # ---- Local training for each agent ----
        result = agent_manager.train(num_iterations=num_iterations)

        # ---- Extract per-agent metrics ----
        round_metrics = agent_manager.get_metrics(result)

        # ---- Log metrics for each agent ----
        for aid in junction_ids:
            m = round_metrics.get(
                aid, {"mean_reward": 0.0, "mean_queue": 0.0, "mean_wait": 0.0}
            )
            log_metrics(
                {
                    "round": r + 1,
                    "agent": aid,
                    "cluster": int(server.cluster_assignments.get(aid, -1)),
                    **m,
                    "status": "trained",
                },
                log_file,
            )

        # ---- Federated clustering and weight aggregation ----
        weights = agent_manager.get_weights()
        server.cluster_agents(weights)  # Assign agents to clusters
        cluster_weights = server.aggregate(weights)  # Aggregate weights per cluster

        # ---- Update agent weights only if changed ----
        for aid, w in cluster_weights.items():
            agent_manager.set_weights({aid: w})

        # ---- Compute average reward for convergence check ----
        avg_reward = sum(
            round_metrics.get(aid, {}).get("mean_reward", 0.0) for aid in junction_ids
        ) / len(junction_ids)
        Logger.success(f"Round {r + 1}/{rounds} | Avg Reward: {avg_reward:.2f}")

        # ---- Early stopping / patience logic ----
        if patience_enabled:
            if avg_reward - best_reward >= min_reward_delta:
                best_reward = avg_reward
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_epochs:
                    Logger.info(
                        f"Training converged early at round {r + 1} (patience limit reached)."
                    )
                    break

        # ---- Save checkpoint periodically ----
        if (r + 1) % save_freq == 0:
            agent_manager.save()

    # ---- Save final checkpoint ----
    final_checkpoint = agent_manager.save()
    Logger.success(f"Training complete | Final checkpoint: {final_checkpoint}")

    # ---- Plot training progress ----
    try:
        plot_training(log_file=log_file, output_dir="plots")
        plot_clusters(log_file=log_file, output_dir="plots")
    except Exception as e:
        Logger.warning(f"Plotting failed: {e}")

    # ---- Cleanup ----
    agent_manager.close()
    env.close()
