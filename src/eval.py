"""
Evaluation entry for CF-MADRL.
Runs simulation and generates plots internally.
"""

import os
import json


import time
import numpy as np
import traci
import sumolib

from src.agent_manager import AgentManager
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from tools.visualize import plot_evaluation
from tools.utils import ensure_dir, Logger, scan_topology


def evaluate_rl(config):

    Logger.header("CF-MADRL Evaluation")

    sim_steps = 100_000
    junction_ids = config["system"]["controlled_junctions"]

    # ---- Discover topology ----
    max_lanes, max_phases = scan_topology(config)

    # ---- Build environment ----
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

    # ---- Load agents ----
    agent_manager = AgentManager(env, config, junction_ids)

    checkpoint_dir = os.path.abspath(config["system"]["model_save_path"])

    if os.path.exists(checkpoint_dir):
        agent_manager.load(checkpoint_dir)
        Logger.success("Checkpoint loaded.")
    else:
        Logger.warning("No checkpoint found. Using random policies.")
        agent_manager.build()

    # ---- Run simulation ----
    obs, _ = env.reset()

    metrics = {aid: {"rewards": [], "queues": [], "waits": []} for aid in junction_ids}

    done = False
    step = 0
    clearance_time = 0

    while step < sim_steps and not done:
        actions = {}

        for aid in junction_ids:
            policy = agent_manager.algo.get_policy(aid)
            action, _, _ = policy.compute_single_action(
                obs[aid],
                explore=False,  # Deterministic evaluation
            )
            actions[aid] = action

        obs, rewards, dones, truncs, infos = env.step(actions)

        for aid in junction_ids:
            metrics[aid]["rewards"].append(rewards[aid])
            metrics[aid]["queues"].append(infos[aid].get("step_queue", 0))
            metrics[aid]["waits"].append(infos[aid].get("step_wait", 0))

        step += 1
        done = any(dones.values()) or all(truncs.values())

        if done:
            clearance_time = env.steps_counter * config["sumo"]["step_length"]

    agent_manager.close()
    env.close()

    # ---- Save logs ----
    base_log_dir = config["system"].get("log_dir", "logs")
    log_dir = os.path.join(base_log_dir, "cfmadrl")
    ensure_dir(log_dir)

    log_data = {
        "cfmadrl": metrics,
        "clearance_time": clearance_time,
    }

    log_path = os.path.join(log_dir, "evaluation_logs.json")

    with open(log_path, "w") as f:
        json.dump(log_data, f)

    Logger.success("Evaluation completed.")
    Logger.info(f"Clearance Time: {clearance_time:.2f}s")

    # ---- Plot inside evaluation ----
    try:
        plot_evaluation(
            log_file=log_path,
            output_dir="plots",
            algo_name="CF-MADRL",
            model_name="cfmadrl",
        )
        Logger.success("Plots generated successfully.")
    except Exception as e:
        Logger.warning(f"Plotting failed: {e}")

    return {
        "metrics": metrics,
        "clearance_time": clearance_time,
    }


def run_fixed_time_baseline(config):
    """
    Run a Fixed-Time traffic light baseline in SUMO using TraCI.

    Args:
        config: dict containing SUMO config, junction IDs, step length, GUI flag

    Returns:
        dict:
            metrics: per junction, containing 'rewards', 'queues', 'waits'
            clearance_time: simulation time when all vehicles cleared
            phase_switches: number of traffic light phase changes per junction
    """

    Logger.section("Running Fixed-Time Baseline Comparison")

    # Decide SUMO binary (GUI or CLI)
    gui = config["evaluation"].get("gui", False)
    sumo_bin = sumolib.checkBinary("sumo-gui" if gui else "sumo")

    # Build SUMO command
    step_length = config["sumo"]["step_length"]
    sumo_cmd = [
        sumo_bin,
        "-c",
        config["sumo"]["config_file"],
        "--step-length",
        str(step_length),
    ]
    if gui and (delay := config["evaluation"].get("step_delay", 0)) > 0:
        sumo_cmd.extend(["--delay", str(int(delay))])
        sumo_cmd.append("--start")

    junctions = config["system"]["controlled_junctions"]
    sim_steps = 100_000  # large number to ensure full simulation

    label = f"fixed_{np.random.randint(999999)}"  # unique label for TraCI connection

    # --- Connect to SUMO with retry logic ---
    conn = None
    for attempt in range(5):
        try:
            Logger.info(
                f"Starting SUMO (Attempt {attempt + 1}/5): {' '.join(sumo_cmd)}"
            )
            traci.start(sumo_cmd, label=label)
            time.sleep(1)
            conn = traci.getConnection(label)
            break
        except Exception as ex:
            if attempt < 4:
                Logger.warning(f"TraCI connection failed: {ex}. Retrying in 2s...")
                try:
                    traci.close()
                except Exception:
                    pass
                time.sleep(2)
            else:
                Logger.error("TraCI connection failed after max retries.")
                raise ex

    # --- Initialize metrics and tracking ---
    metrics = {j: {"rewards": [], "queues": [], "waits": []} for j in junctions}
    clearance_time = 0
    phase_switches = {j: 0 for j in junctions}
    last_phase = {j: None for j in junctions}

    # Switch all junctions to fixed-time program
    for j_id in junctions:
        try:
            conn.trafficlight.setProgram(j_id, "fixed")
        except Exception:
            pass

    Logger.info("Running Fixed-Time baseline simulation...")

    # --- Main simulation loop ---
    step = 0
    while step < sim_steps:
        remaining_vehicles = conn.simulation.getMinExpectedNumber()
        if remaining_vehicles <= 0:
            clearance_time = conn.simulation.getTime()
            Logger.info(f"Fixed-Time: All vehicles cleared at {clearance_time:.1f}s")
            break

        conn.simulationStep()

        for j_id in junctions:
            # Track phase switches
            current_phase = conn.trafficlight.getPhase(j_id)
            if last_phase[j_id] is not None and current_phase != last_phase[j_id]:
                phase_switches[j_id] += 1
            last_phase[j_id] = current_phase

            # Compute queues and waits for controlled lanes
            lanes = list(set(conn.trafficlight.getControlledLanes(j_id)))
            total_queue = sum(conn.lane.getLastStepHaltingNumber(ln) for ln in lanes)
            total_wait = sum(conn.lane.getWaitingTime(ln) for ln in lanes)

            # Reward: negative sum of queue + wait (normalized by lane count)
            metrics[j_id]["rewards"].append(-(total_queue + total_wait) / len(lanes))
            metrics[j_id]["queues"].append(total_queue)
            metrics[j_id]["waits"].append(total_wait)

        step += 1

    # --- Close SUMO safely ---
    if conn:
        try:
            conn.close()
        except Exception:
            pass
    time.sleep(5)  # ensure ports are released

    return {
        "metrics": metrics,
        "clearance_time": clearance_time,
        "phase_switches": phase_switches,
    }
