"""
Training loop for DQN multi-agent traffic signal control.

Runs one independent DQNAgent per junction, step-by-step through MultiAgentSumoEnv.
Saves logs to logs/dqn/training_logs.json and plots to plots/dqn/.
"""

import os
import warnings
import logging
import torch

from src.dqn_agent import DQNAgent
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from tools.utils import ensure_dir, log_metrics, Logger, scan_topology
from tools.visualize import plot_training


def train_dqn(config, args_rounds=None):
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)

    Logger.header("DQN Multi-Agent Training")

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    Logger.info(f"Training Device: {device_name.upper()}")

    # ------------------------------------------------
    # Extract Config Sections (Single Source of Truth)
    # ------------------------------------------------
    dqn_cfg = config["rl"]["dqn"]
    train_cfg = config["training"]
    system_cfg = config["system"]

    # ------------------------------------------------
    # Directories
    # ------------------------------------------------
    model_dir = "saved_models/dqn"
    log_dir = "logs/dqn"
    plot_dir = "plots/dqn"
    log_file = os.path.join(log_dir, "training_logs.json")

    ensure_dir(model_dir)
    ensure_dir(log_dir)
    ensure_dir(plot_dir)

    # ------------------------------------------------
    # Training Parameters
    # ------------------------------------------------
    local_steps = train_cfg["local_steps_per_round"]
    rounds = args_rounds if args_rounds is not None else train_cfg["federated_rounds"]
    save_freq = train_cfg["save_freq"]

    # ------------------------------------------------
    # 1️⃣ Discover Environment Topology
    # ------------------------------------------------
    Logger.section("Phase 1: Environment Discovery & Setup")

    max_lanes, max_phases = scan_topology(config)
    junction_ids = system_cfg["controlled_junctions"]

    Logger.narrative(
        f"Topology: {max_lanes} lanes, {max_phases} phases, {len(junction_ids)} junctions"
    )

    # ------------------------------------------------
    # 2️⃣ Initialize Environment
    # ------------------------------------------------
    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
        }
    )

    obs, _ = env.reset()

    first_aid = junction_ids[0]
    n_actions = env.action_space[first_aid].n
    obs_dim = env.observation_space[first_aid].shape[0]

    # ------------------------------------------------
    # 3️⃣ Create Agents (Fully Config-Driven)
    # ------------------------------------------------
    agents = {
        aid: DQNAgent(
            junction_id=aid,
            obs_dim=obs_dim,
            n_actions=n_actions,
            lr=dqn_cfg["lr"],
            gamma=dqn_cfg["gamma"],
            epsilon=dqn_cfg["epsilon"],
            epsilon_min=dqn_cfg["epsilon_min"],
            epsilon_decay=dqn_cfg["epsilon_decay"],
            batch_size=dqn_cfg["batch_size"],
            buffer_size=dqn_cfg["buffer_size"],
            target_update_freq=dqn_cfg["target_update_freq"],
            fcnet_hiddens=dqn_cfg["fcnet_hiddens"],
        )
        for aid in junction_ids
    }

    # ------------------------------------------------
    # Load Existing Checkpoints
    # ------------------------------------------------
    for aid in junction_ids:
        ckpt = os.path.join(model_dir, f"{aid}.pt")
        if os.path.exists(ckpt):
            agents[aid].load(ckpt)
            agents[aid].epsilon = dqn_cfg["epsilon"]
            Logger.info(f"Loaded DQN weights for {aid}")

    if os.path.exists(log_file):
        os.remove(log_file)

    # ------------------------------------------------
    # 4️⃣ Training Loop
    # ------------------------------------------------
    for r in range(rounds):
        current_round = r + 1

        obs, _ = env.reset()
        done = False
        step = 0

        round_rewards = {aid: [] for aid in junction_ids}
        round_losses = {aid: [] for aid in junction_ids}
        round_queues = {aid: [] for aid in junction_ids}
        round_waits = {aid: [] for aid in junction_ids}

        while step < local_steps and not done:
            actions = {aid: agents[aid].select_action(obs[aid]) for aid in junction_ids}
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)

            is_done = terminateds.get("__all__", False)

            for aid in junction_ids:
                agents[aid].store(
                    obs[aid], actions[aid], rewards[aid], next_obs[aid], float(is_done)
                )

                loss = agents[aid].train_step()

                round_rewards[aid].append(rewards[aid])
                round_queues[aid].append(infos[aid].get("step_queue", 0.0))
                round_waits[aid].append(infos[aid].get("step_wait", 0.0))

                if loss > 0:
                    round_losses[aid].append(loss)

            obs = next_obs
            done = is_done
            step += 1

        # ------------------------------------------------
        # Log Metrics
        # ------------------------------------------------
        for aid in junction_ids:
            mean_reward = sum(round_rewards[aid]) / max(len(round_rewards[aid]), 1)
            mean_loss = (
                sum(round_losses[aid]) / len(round_losses[aid])
                if round_losses[aid]
                else 0.0
            )
            mean_queue = sum(round_queues[aid]) / max(len(round_queues[aid]), 1)
            mean_wait = sum(round_waits[aid]) / max(len(round_waits[aid]), 1)

            log_metrics(
                {
                    "round": current_round,
                    "agent": aid,
                    "mean_reward": mean_reward,
                    "mean_loss": mean_loss,
                    "mean_queue": mean_queue,
                    "mean_wait": mean_wait,
                    "epsilon": agents[aid].epsilon,
                    "status": "trained",
                },
                log_file,
            )

        # ------------------------------------------------
        # Decay Exploration + Learning Rate
        # ------------------------------------------------
        for aid in junction_ids:
            agents[aid].epsilon = max(
                agents[aid].epsilon_min,
                agents[aid].epsilon * dqn_cfg["epsilon_decay"],
            )

            min_lr = dqn_cfg["lr"] * 1e-3
            for param_group in agents[aid].optimizer.param_groups:
                param_group["lr"] = max(min_lr, param_group["lr"] * dqn_cfg["lr_decay"])

        avg_reward = sum(
            sum(round_rewards[aid]) / max(len(round_rewards[aid]), 1)
            for aid in junction_ids
        ) / len(junction_ids)

        Logger.success(
            f"Round {r + 1}/{rounds} | Avg Reward: {avg_reward:.2f} | ε: {agents[first_aid].epsilon:.4f}"
        )

        if current_round % save_freq == 0:
            for aid in junction_ids:
                agents[aid].save(os.path.join(model_dir, f"{aid}.pt"))
            Logger.info(f"Checkpoint saved at round {current_round}")

    # Final save
    for aid in junction_ids:
        agents[aid].save(os.path.join(model_dir, f"{aid}.pt"))

    Logger.success("Training complete. Models saved.")

    plot_training(log_file=log_file, output_dir=plot_dir, algo_name="DQN")

    env.close()
