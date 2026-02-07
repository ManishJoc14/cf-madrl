"""
Training entry for CF-MADRL using RLlib multi-agent PPO.
"""

import os
import json
import logging
import warnings

from src.agent_manager import AgentManager
from src.federation import ClusteredFederatedServer
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from tools.visualize import plot_training, plot_clusters
from tools.utils import ensure_dir, log_metrics, Logger, scan_topology


def train_rllib(config, args_rounds=None):
    """
    Train CF-MADRL agents using RLlib multi-agent PPO in a shared SUMO environment.

    Steps:
    1. Discover SUMO network topology (max lanes & phases).
    2. Initialize Multi-Agent SUMO environment.
    3. Setup RLlib AgentManager for each junction.
    4. Setup FederatedServer for clustering and aggregation.
    5. Run federated training loop:
        a) Local training
        b) Metrics logging
        c) Clustering of agents
        d) Weight aggregation
        e) Distribute updated models
    6. Save final model and print cluster summary.
    """

    # Silence noisy loggers
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)

    Logger.header("CF-MADRL Federated Training")

    # Setup directories
    ensure_dir(config["system"]["model_save_path"])
    ensure_dir("logs")

    # Training parameters
    local_steps = config["training"]["local_steps_per_round"]
    rounds = (
        args_rounds
        if args_rounds is not None
        else config["training"]["federated_rounds"]
    )
    save_freq = config["training"]["save_freq"]
    n_clusters = config["training"]["n_clusters"]

    # Check for GPU
    import torch

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    Logger.info(f"Training Device Detected: {device_name.upper()}")
    if device_name == "cuda":
        Logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Scan SUMO topology
    Logger.section("Phase 1: Environment Discovery & Setup")
    Logger.narrative("Scanning SUMO network topology...")
    max_lanes, max_phases = scan_topology(config)
    junction_ids = config["system"]["controlled_junctions"]
    Logger.narrative(
        f"Topology discovered: {max_lanes} lanes, {max_phases} phases across {len(junction_ids)} junctions."
    )

    # 2. Initialize environment
    Logger.narrative("Initializing Multi-Agent SUMO Environment...")
    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
        }
    )

    # 3. Setup RLlib AgentManager
    checkpoint_dir = os.path.abspath(config["system"]["model_save_path"])
    is_resuming = os.path.exists(checkpoint_dir) and any(os.scandir(checkpoint_dir))
    Logger.narrative("Building RLlib PPO Algorithm Stack...")
    agent_manager = AgentManager(env, config, junction_ids)

    if is_resuming:
        Logger.info(f"Resuming training from checkpoint: {checkpoint_dir}")
        try:
            agent_manager.load(checkpoint_dir)
            Logger.success("Checkpoint loaded successfully.")
        except Exception as e:
            Logger.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            agent_manager.build()
    else:
        agent_manager.build()

    # 4. Setup ClusteredFederatedServer
    server = ClusteredFederatedServer(n_clusters=n_clusters)
    Logger.success(
        f"System ready | Agents: {len(junction_ids)}, Clusters: {n_clusters}, Rounds: {rounds}"
    )

    # 5. Training Loop
    log_file = os.path.join(
        config["system"].get("log_dir", "logs"), "training_logs.json"
    )
    if not is_resuming and os.path.exists(log_file):
        os.remove(log_file)

    start_round = 1
    if is_resuming and os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
                if logs:
                    start_round = max(ln["round"] for ln in logs) + 1
        except Exception as ex:
            Logger.warning(f"Could not determine start round from logs: {ex}")

    Logger.info(f"Starting training from Round {start_round}")

    for r in range(rounds):
        current_round = start_round + r
        Logger.round_banner(current_round, rounds + start_round - 1)

        # a) Local Training
        Logger.narrative(f"Phase 1: Local Training for {len(junction_ids)} agents...")
        batch_size = config["rl"].get("train_batch_size", 512)
        num_iterations = max(1, local_steps // batch_size)
        result = agent_manager.train(num_iterations=num_iterations)

        # b) Metrics Logging
        Logger.narrative("Phase 2: Logging metrics for each agent...")
        round_metrics = agent_manager.get_metrics(result)
        display_reward = 0.0
        active_agents = 0
        for aid in junction_ids:
            if aid in round_metrics:
                display_reward += round_metrics[aid]["mean_reward"]
                active_agents += 1
        display_reward = display_reward / active_agents if active_agents > 0 else 0.0

        # Save metrics to log file
        for aid in junction_ids:
            m = round_metrics.get(
                aid, {"mean_reward": 0.0, "mean_queue": 0.0, "mean_wait": 0.0}
            )
            log_metrics(
                {
                    "round": current_round,
                    "agent": aid,
                    "status": "trained",
                    "cluster": int(server.cluster_assignments.get(aid, -1)),
                    **m,
                },
                log_file,
            )

        # c) Clustering
        Logger.narrative("Phase 3: Clustering agents based on weights...")
        agent_weights = agent_manager.get_weights()
        server.cluster_agents(agent_weights)

        # d) Federated Aggregation
        Logger.narrative("Phase 4: Aggregating weights within clusters...")
        cluster_weights = server.aggregate(agent_weights)

        # e) Redistribute models
        Logger.narrative("Phase 5: Updating agents with cluster-aggregated weights...")
        for aid, weights in cluster_weights.items():
            agent_manager.set_weights({aid: weights})

        Logger.success(
            f"Round {current_round}/{rounds} complete | Avg Reward: {display_reward:7.2f}"
        )

        # Save checkpoint periodically
        if current_round % save_freq == 0:
            agent_manager.save()
            # Save norm stats as well
            norm_path = os.path.join(
                config["system"].get("log_dir", "logs"), "norm_stats.json"
            )
            env.save_norm_stats(norm_path)

    # 6. Final save & cluster summary
    final_checkpoint = agent_manager.save()
    Logger.success(f"Training complete | Final checkpoint: {final_checkpoint}")

    Logger.section("Final Federated Cluster Summary")
    cluster_groups = {}
    for aid, cid in server.cluster_assignments.items():
        cluster_groups.setdefault(cid, []).append(aid)

    for cid, agents in sorted(cluster_groups.items()):
        print(f"Cluster {cid}:")
        for aid in agents:
            print(f"  - {aid}")
    print("-" * 50)

    # Generate Training Plots
    Logger.narrative("Generating training plots...")
    try:
        # generate reward/queue/cluster plots
        plot_training(log_file="logs/training_logs.json", output_dir="plots")
        plot_clusters(log_file="logs/training_logs.json", output_dir="plots")
        Logger.success("Training plots generated in 'plots/' directory.")
    except Exception as e:
        Logger.warning(f"Failed to generate training plots: {e}")

    # Cleanup
    agent_manager.close()
    norm_path = os.path.join(config["system"].get("log_dir", "logs"), "norm_stats.json")
    env.save_norm_stats(norm_path)
    env.close()
