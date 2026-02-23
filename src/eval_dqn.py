"""
Evaluation for DQN multi-agent traffic signal control.
"""

import os
import json

from src.dqn_agent import DQNAgent
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from src.eval import run_fixed_time_baseline
from tools.utils import ensure_dir, Logger, scan_topology, _resample_metrics_to_length
from tools.visualize import plot_evaluation


def evaluate_dqn(config):
    Logger.header("DQN Agent Evaluation")

    dqn_cfg = config["rl"]["dqn"]
    eval_cfg = config["evaluation"]
    system_cfg = config["system"]

    model_dir = "saved_models/dqn"
    log_dir = system_cfg["log_dir"]
    plot_dir = "plots/dqn"

    ensure_dir(plot_dir)

    sim_steps = eval_cfg["steps"]
    junction_ids = system_cfg["controlled_junctions"]

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
            "step_delay": eval_cfg["step_delay"],
        }
    )

    obs, _ = env.reset()

    first_aid = junction_ids[0]
    n_actions = env.action_space[first_aid].n
    obs_dim = env.observation_space[first_aid].shape[0]

    agents = {}

    for aid in junction_ids:
        agent = DQNAgent(
            junction_id=aid,
            obs_dim=obs_dim,
            n_actions=n_actions,
            epsilon=0.0,
            fcnet_hiddens=dqn_cfg["fcnet_hiddens"],
        )

        ckpt = os.path.join(model_dir, f"{aid}.pt")
        if os.path.exists(ckpt):
            agent.load(ckpt)
            agent.epsilon = 0.0
            Logger.success(f"Loaded DQN model for {aid}")

        agents[aid] = agent

    dqn_metrics = {
        aid: {"rewards": [], "queues": [], "waits": []} for aid in junction_ids
    }

    step = 0
    done = False
    clearance_time = 0

    while step < sim_steps and not done:
        actions = {aid: agents[aid].select_action(obs[aid]) for aid in junction_ids}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for aid in junction_ids:
            dqn_metrics[aid]["rewards"].append(rewards[aid])
            dqn_metrics[aid]["queues"].append(infos[aid].get("step_queue", 0))
            dqn_metrics[aid]["waits"].append(infos[aid].get("step_wait", 0))

        step += 1
        done = terminateds.get("__all__", False)

    if done:
        clearance_time = env.steps_counter * config["sumo"]["step_length"]

    env.close()

    fixed_metrics = run_fixed_time_baseline(config)

    target_len = min(len(dqn_metrics[aid]["rewards"]) for aid in junction_ids)
    fixed_aligned = _resample_metrics_to_length(fixed_metrics["metrics"], target_len)

    log_data = {
        "dqn": dqn_metrics,
        "fixed_time": fixed_aligned,
        "clearance_times": {
            "dqn": clearance_time,
            "fixed_time": fixed_metrics["clearance_time"],
        },
    }

    ensure_dir(log_dir)
    eval_log = os.path.join(log_dir, "evaluation_logs_dqn.json")

    with open(eval_log, "w") as f:
        json.dump(log_data, f)

    plot_evaluation(log_file=eval_log, output_dir=plot_dir, algo_name="DQN", model_name="dqn")

    return {
        "metrics": dqn_metrics,
        "clearance_time": clearance_time,
    }
