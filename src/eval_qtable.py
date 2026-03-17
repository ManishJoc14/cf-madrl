"""
Evaluation for Q-Table multi-agent traffic signal control.
"""

import os
import json

from src.qtable_agent import QTableAgent
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from src.eval import run_fixed_time_baseline
from tools.utils import ensure_dir, Logger, scan_topology, _resample_metrics_to_length
from tools.visualize import plot_evaluation


def evaluate_qtable(config):

    Logger.header("Q-Table Agent Evaluation")

    model_dir = "saved_models/qtable"
    base_log_dir = config["system"].get("log_dir", "logs")
    log_dir = os.path.join(base_log_dir, "qtable")
    plot_dir = "plots/qtable"
    ensure_dir(plot_dir)

    sim_steps = 100_000
    junction_ids = config["system"]["controlled_junctions"]

    qcfg = config["rl"]["qtable"]

    # Environment
    max_lanes, max_phases = scan_topology(config)

    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
            "max_steps": sim_steps,
            "evaluation_mode": True,
        }
    )

    obs, _ = env.reset()
    first_aid = junction_ids[0]
    n_actions = env.action_space[first_aid].n

    # Load Agents (Greedy)
    agents = {}

    for aid in junction_ids:
        agent = QTableAgent(
            junction_id=aid,
            n_actions=n_actions,
            epsilon=0.0,
            n_bins=qcfg["n_bins"],
            bin_edges=qcfg["bin_edges"],
        )

        ckpt = os.path.join(model_dir, f"{aid}.pkl")

        if os.path.exists(ckpt):
            agent.load(ckpt)
            agent.epsilon = 0.0
            Logger.success(f"Loaded Q-table for {aid}")
        else:
            Logger.warning(f"No checkpoint found for {aid}")

        agents[aid] = agent

    # Evaluation Loop
    qtable_metrics = {
        aid: {"rewards": [], "queues": [], "waits": []} for aid in junction_ids
    }

    step = 0
    done = False
    clearance_time = 0

    while step < sim_steps and not done:
        actions = {aid: agents[aid].select_action(obs[aid]) for aid in junction_ids}

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for aid in junction_ids:
            qtable_metrics[aid]["rewards"].append(rewards[aid])
            qtable_metrics[aid]["queues"].append(infos[aid].get("step_queue", 0))
            qtable_metrics[aid]["waits"].append(infos[aid].get("step_wait", 0))

        step += 1
        done = terminateds.get("__all__", False)

    if done:
        clearance_time = env.steps_counter * config["sumo"]["step_length"]

    env.close()

    # Fixed-Time Baseline
    fixed_metrics = run_fixed_time_baseline(config)

    # Align lengths
    target_len = min(len(qtable_metrics[aid]["rewards"]) for aid in junction_ids)
    fixed_aligned = _resample_metrics_to_length(fixed_metrics["metrics"], target_len)

    # Save logs
    log_data = {
        "qtable": qtable_metrics,
        "fixed_time": fixed_aligned,
        "clearance_times": {
            "qtable": clearance_time,
            "fixed_time": fixed_metrics["clearance_time"],
        },
    }

    ensure_dir(log_dir)
    eval_log = os.path.join(log_dir, "evaluation_logs.json")

    with open(eval_log, "w") as f:
        json.dump(log_data, f)

    Logger.success(f"Evaluation logs saved to {eval_log}")

    # Plots
    try:
        plot_evaluation(log_file=eval_log, output_dir=plot_dir, algo_name="Q-Table", model_name='qtable')
    except Exception as e:
        Logger.warning(f"Plotting failed: {e}")

    return {
        "metrics": qtable_metrics,
        "clearance_time": clearance_time,
    }
