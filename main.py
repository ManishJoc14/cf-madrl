"""
Main script for CF-MADRL training using RLlib multi-agent PPO.
All agents train in a single shared SUMO simulation.
"""

import os
import json
import time
import traci
import logging
import sumolib
import argparse
import warnings
import numpy as np
from agent_manager import AgentManager
from federation import ClusteredFederatedServer
from multi_agent_sumo_env import MultiAgentSumoEnv
from visualize import plot_training, plot_clusters, plot_evaluation
from utils import load_config, ensure_dir, log_metrics, Logger, scan_topology

# Global silencing
warnings.filterwarnings("ignore")
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("ray.rllib").setLevel(logging.ERROR)


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


def run_fixed_time_baseline(config):
    """
    Run the Fixed-Time traffic light baseline in SUMO.

    Returns:
        metrics: dict
            Structure:
            {
                "Junction_1": {"rewards": [...], "queues": [...], "waits": [...]},
                "Junction_2": {...},
            }
    """

    Logger.section("Running Fixed-Time Baseline Comparison")

    # Decide SUMO binary (GUI or CLI)
    gui = config["evaluation"].get("gui", False)
    sumo_bin = sumolib.checkBinary("sumo-gui" if gui else "sumo")

    # Build SUMO command
    sumo_cmd = [
        sumo_bin,
        "-c",
        config["sumo"]["config_file"],
        "--step-length",
        str(config["sumo"]["step_length"]),
    ]
    if gui:
        sumo_cmd.append("--start")  # Only needed for GUI mode

    # Junctions to control
    junctions = config["system"]["controlled_junctions"]

    # Large number to ensure simulation runs completely
    sim_steps = 100_000

    # Unique label for SUMO instance
    label = f"fixed_{np.random.randint(999999)}"

    # Retry logic to safely start TraCI
    max_retries = 5
    conn = None
    for attempt in range(max_retries):
        try:
            Logger.info(
                f"Starting SUMO (Attempt {attempt + 1}/{max_retries}): {' '.join(sumo_cmd)}"
            )
            traci.start(sumo_cmd, label=label)
            time.sleep(1)
            conn = traci.getConnection(label)
            break
        except Exception as ex:
            if attempt < max_retries - 1:
                Logger.warning(f"TraCI connection failed: {ex}. Retrying in 2s...")
                try:
                    traci.close()
                except Exception:
                    pass
                time.sleep(2)
            else:
                Logger.error(f"TraCI connection failed after {max_retries} attempts.")
                raise ex

    # Initialize metrics dictionary
    metrics = {j: {"rewards": [], "queues": [], "waits": []} for j in junctions}

    try:
        # Main simulation loop
        for _ in range(sim_steps):
            if conn.simulation.getMinExpectedNumber() <= 0:
                break  # Stop when no vehicles left
            conn.simulationStep()

            # Collect rewards, queue lengths, and waiting times
            for j_id in junctions:
                lanes = list(set(conn.trafficlight.getControlledLanes(j_id)))
                total_queue = sum(
                    conn.lane.getLastStepHaltingNumber(ln) for ln in lanes
                )
                total_wait = sum(conn.lane.getWaitingTime(ln) for ln in lanes)

                # Reward: negative sum of queue + wait time (Normalized per lane)
                metrics[j_id]["rewards"].append(
                    -(total_queue + total_wait) / len(lanes)
                )
                metrics[j_id]["queues"].append(total_queue)
                metrics[j_id]["waits"].append(total_wait)

    finally:
        # Close SUMO safely
        if conn:
            try:
                conn.close()
            except:
                pass
        time.sleep(5)  # Wait to release ports

    return metrics


def evaluate_rllib(config):
    """
    Evaluate trained RLlib MADRL agents against the Fixed-Time baseline.

    Steps:
    1. Run Fixed-Time baseline simulation.
    2. Run RLlib MADRL agents in the same SUMO environment.
    3. Collect metrics: rewards, queues, waiting times.
    4. Save logs and generate evaluation plots.
    """

    Logger.header("CF-MADRL Performance Evaluation")

    # 1. Fixed-Time baseline metrics
    fixed_metrics = run_fixed_time_baseline(config)
    # fixed_metrics structure:
    # {
    #   "Junction_1": {"rewards": [...], "queues": [...], "waits": [...]},
    #   "Junction_2": {...},
    # }

    # 2. RLlib MADRL metrics
    # Discover SUMO topology for max lanes/phases
    max_lanes, max_phases = scan_topology(config)
    sim_steps = 100_000  # Large number to ensure simulation runs completely
    junction_ids = config["system"]["controlled_junctions"]

    # Initialize multi-agent SUMO environment
    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
            "max_steps": sim_steps,
        }
    )

    # Setup RLlib agent manager
    agent_manager = AgentManager(env, config, junction_ids)
    checkpoint_dir = os.path.abspath(config["system"]["model_save_path"])
    if os.path.exists(checkpoint_dir):
        Logger.info(f"Loading trained checkpoint from {checkpoint_dir}...")
        try:
            agent_manager.load(checkpoint_dir)
        except Exception as e:
            Logger.warning(f"Failed to load checkpoint: {e}")
            agent_manager.build()
    else:
        Logger.info("No checkpoint found. Using randomly initialized policies.")
        agent_manager.build()

    # Prepare metrics container
    madrl_metrics = {
        aid: {"rewards": [], "queues": [], "waits": []} for aid in junction_ids
    }

    # Reset environment
    obs, _ = env.reset()
    step = 0
    done = False

    # Main simulation loop
    while step < sim_steps and not done:
        actions = {}
        # Compute action for each agent
        for aid in junction_ids:
            policy = agent_manager.algo.get_policy(aid)
            action, _, _ = policy.compute_single_action(obs[aid])
            actions[aid] = action

            # Metadata for duration bias check
            meta = env.junction_metadata[aid]
            num_g = meta["num_green"]
            dur_idx = action % env.num_durations
            green_idx_local = (action // env.num_durations) % num_g
            _duration_val = env.durations[min(dur_idx, env.num_durations - 1)]
            _target_p = meta["green_phases"][green_idx_local]

            # print(f"Agent {aid} | Action {action} -> Phase {_target_p}, Dur {_duration_val}")

        # Step environment
        obs, rewards, dones, truncs, infos = env.step(actions)

        # Collect metrics
        for aid in junction_ids:
            madrl_metrics[aid]["rewards"].append(rewards[aid])
            madrl_metrics[aid]["queues"].append(infos[aid].get("step_queue", 0))
            madrl_metrics[aid]["waits"].append(infos[aid].get("step_wait", 0))

        step += 1
        done = any(dones.values()) or all(truncs.values())

    # 3. Save combined evaluation logs
    log_data = {"madrl": madrl_metrics, "fixed_time": fixed_metrics}
    ensure_dir("logs")
    with open("logs/evaluation_logs.json", "w") as f:
        json.dump(log_data, f)
    Logger.success("Evaluation logs saved to logs/evaluation_logs.json")

    # 4. Generate evaluation plots
    try:
        plot_evaluation(log_file="logs/evaluation_logs.json", output_dir="plots")
    except Exception as e:
        Logger.warning(f"Plotting failed: {e}")

    # 5. Print summary table
    Logger.section("Evaluation Summary (Average Reward)")
    print(f"{'Junction ID':<35} | {'MADRL':>10} | {'Fixed':>10} | {'Improvement':>10}")
    print("-" * 80)

    for aid in junction_ids:
        r_madrl = np.mean(madrl_metrics[aid]["rewards"])
        r_fixed = np.mean(fixed_metrics[aid]["rewards"])

        # Since rewards are negative (penalty), higher (closer to 0) is better
        abs_madrl = abs(r_madrl)
        abs_fixed = abs(r_fixed)
        improvement = (
            ((abs_fixed - abs_madrl) / abs_fixed * 100) if abs_fixed > 0 else 0.0
        )

        print(
            f"{aid[:35]:<35} | {r_madrl:10.2f} | {r_fixed:10.2f} | {improvement:9.1f}%"
        )
    print("-" * 80)

    # Cleanup
    agent_manager.close()
    norm_path = os.path.join(config["system"].get("log_dir", "logs"), "norm_stats.json")
    env.save_norm_stats(norm_path)
    env.close()


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
        Logger.section("Launching Evaluation Only")
        evaluate_rllib(config)
