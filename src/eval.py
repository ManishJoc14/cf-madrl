"""
Evaluation entry for CF-MADRL against Fixed-Time baseline.
"""

import os
import sys
import json
import time
import traci
import sumolib
import numpy as np

# Import
from src.agent_manager import AgentManager
from src.multi_agent_sumo_env import MultiAgentSumoEnv
from src import federation
from tools.visualize import plot_evaluation
from tools.utils import ensure_dir, Logger, scan_topology, _resample_metrics_to_length
import tools.utils as utils_module
import tools.visualize as visualize_module

# This allows old checkpoints saved with "agent_manager" imports to load with "src.agent_manager"
sys.modules['agent_manager'] = sys.modules['src.agent_manager']
sys.modules['multi_agent_sumo_env'] = sys.modules['src.multi_agent_sumo_env']
sys.modules['federation'] = sys.modules['src.federation']
sys.modules['utils'] = utils_module
sys.modules['visualize'] = visualize_module


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
    step_delay = config["evaluation"].get("step_delay", 0)
    sumo_cmd = [
        sumo_bin,
        "-c",
        config["sumo"]["config_file"],
        "--step-length",
        str(config["sumo"]["step_length"]),
    ]
    if gui:
        sumo_cmd.append("--start")  # Start simulation immediately in GUI
        # Add SUMO's native delay parameter (in milliseconds)
        if step_delay > 0:
            sumo_cmd.extend(["--delay", str(int(step_delay))])

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
                Logger.error("TraCI connection failed after max retries.")
                raise ex

    # Initialize metrics dictionary
    metrics = {j: {"rewards": [], "queues": [], "waits": []} for j in junctions}
    clearance_time = 0
    
    # Track phase switches for fixed-time
    phase_switches = {j: 0 for j in junctions}
    last_phase = {j: None for j in junctions}

    # Switch all junctions to the fixed-time program
    for j_id in junctions:
        try:
            conn.trafficlight.setProgram(j_id, "fixed")
        except Exception:
            pass

    Logger.info("Running Fixed-Time baseline.")

    try:
        # Main simulation loop - run until all vehicles are cleared
        step = 0
        while step < sim_steps:
            # Check if simulation is complete (no vehicles left)
            remaining_vehicles = conn.simulation.getMinExpectedNumber()
            if remaining_vehicles <= 0:
                clearance_time = conn.simulation.getTime()
                Logger.info(f"Fixed-Time: All vehicles cleared at {clearance_time:.1f}s")
                break

            conn.simulationStep()

            # Collect rewards, queue lengths, and waiting times
            for j_id in junctions:
                # Track phase switches
                current_phase = conn.trafficlight.getPhase(j_id)
                if last_phase[j_id] is not None and current_phase != last_phase[j_id]:
                    phase_switches[j_id] += 1
                last_phase[j_id] = current_phase
                
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

            step += 1

    finally:
        # Close SUMO safely
        if conn:
            try:
                conn.close()
            except Exception:
                pass
        time.sleep(5)  # Wait to release ports

    return {"metrics": metrics, "clearance_time": clearance_time, "phase_switches": phase_switches}


def evaluate_rl(config):
    """
    Run RL agents in SUMO evaluation mode.

    Returns:
        madrl_metrics: dict
            Structure:
            {
                "Junction_1": {"rewards": [...], "queues": [...], "waits": [...]},
                "Junction_2": {...},
            }
    """

    Logger.header("CF-MADRL Agent Evaluation")

    # RLlib MADRL metrics
    # Discover SUMO topology for max lanes/phases
    max_lanes, max_phases = scan_topology(config)
    sim_steps = 100_000  # Large number to ensure simulation runs completely
    junction_ids = config["system"]["controlled_junctions"]

    # Initialize multi-agent SUMO environment (with evaluation mode enabled)
    env = MultiAgentSumoEnv(
        {
            "config": config,
            "junction_ids": junction_ids,
            "max_lanes": max_lanes,
            "max_phases": max_phases,
            "max_steps": sim_steps,
            "evaluation_mode": True,  # Enable evaluation GUI and delay settings
        }
    )

    # Load junction-specific policies from RLlib checkpoint
    agent_manager = AgentManager(env, config, junction_ids)
    checkpoint_dir = os.path.abspath(config["system"]["model_save_path"])
    
    if os.path.exists(checkpoint_dir):
        Logger.info(f"Loading junction-specific policies from {checkpoint_dir}...")
        try:
            agent_manager.load(checkpoint_dir)
            Logger.success(f"Loaded {len(junction_ids)} junction-specific policies")
        except Exception as e:
            Logger.error(f"Failed to load checkpoint: {e}")
            Logger.warning("Initializing random policies instead...")
            agent_manager.build()
    else:
        Logger.warning(f"No checkpoint found at {checkpoint_dir}")
        Logger.warning("Initializing random policies...")
        agent_manager.build()

    # Prepare metrics container
    madrl_metrics = {
        aid: {"rewards": [], "queues": [], "waits": []} for aid in junction_ids
    }
    clearance_time = 0
    
    # Track action/duration choices for analysis
    action_stats = {
        aid: {"durations": [], "decisions": 0, "total_duration": 0} 
        for aid in junction_ids
    }

    # Get step delay from config
    step_delay = config["evaluation"].get("step_delay", 0)
    Logger.info("Running CF-MADRL evaluation with junction-specific policies.")

    # Reset environment
    obs, _ = env.reset()
    step = 0
    done = False

    # Main simulation loop - run until all vehicles are cleared
    while step < sim_steps and not done:
        actions = {}
        # Compute action for each agent using its own policy (deterministic)
        for aid in junction_ids:
            policy = agent_manager.algo.get_policy(aid)
            # Use explore=False for deterministic action selection (argmax, no sampling)
            action, _, _ = policy.compute_single_action(obs[aid], explore=False)
            actions[aid] = action
            
            # Decode action to extract duration choice
            meta = env.junction_metadata[aid]
            num_g = meta["num_green"]
            dur_idx = action % env.num_durations
            duration_val = env.durations[min(dur_idx, env.num_durations - 1)]
            
            # Track duration statistics
            action_stats[aid]["durations"].append(duration_val)
            action_stats[aid]["decisions"] += 1
            action_stats[aid]["total_duration"] += duration_val

        # Step environment
        obs, rewards, dones, truncs, infos = env.step(actions)

        # Collect metrics
        for aid in junction_ids:
            madrl_metrics[aid]["rewards"].append(rewards[aid])
            madrl_metrics[aid]["queues"].append(infos[aid].get("step_queue", 0))
            madrl_metrics[aid]["waits"].append(infos[aid].get("step_wait", 0))

        step += 1
        done = any(dones.values()) or all(truncs.values())

        # Check if all vehicles are cleared
        if done:
            clearance_time = env.steps_counter * config["sumo"]["step_length"]
            Logger.info(f"CF-MADRL: All vehicles cleared at {clearance_time:.1f}s")

    # Cleanup
    agent_manager.close()
    norm_path = os.path.join(config["system"].get("log_dir", "logs"), "norm_stats.json")
    env.save_norm_stats(norm_path)
    env.close()

    return {"metrics": madrl_metrics, "clearance_time": clearance_time, "action_stats": action_stats}


def generate_comparison_plots(fixed_metrics, madrl_metrics, config):
    """
    Compare Fixed-Time baseline against CF-MADRL evaluation results.

    Steps:
    1. Auto-align metric series lengths for fair comparison.
    2. Save combined logs.
    3. Generate comparison plots.
    4. Print summary statistics.
    """

    Logger.header("Evaluation Results: Baseline vs CF-MADRL")

    junction_ids = config["system"]["controlled_junctions"]

    # Extract metrics and clearance times
    fixed_time_metrics = fixed_metrics["metrics"]
    madrl_time_metrics = madrl_metrics["metrics"]
    fixed_clearance = fixed_metrics["clearance_time"]
    madrl_clearance = madrl_metrics["clearance_time"]
    # 1. Auto-align metrics to same length
    target_len = min(len(madrl_time_metrics[aid]["rewards"]) for aid in junction_ids)
    Logger.info(
        f"Auto-aligning Fixed-Time metrics to CF-MADRL length: {target_len}"
    )
    fixed_time_metrics = _resample_metrics_to_length(fixed_time_metrics, target_len)

    # 2. Save combined evaluation logs
    log_data = {
        "madrl": madrl_time_metrics, 
        "fixed_time": fixed_time_metrics,
        "clearance_times": {"madrl": madrl_clearance, "fixed_time": fixed_clearance}
    }
    ensure_dir("logs")
    with open("logs/evaluation_logs.json", "w") as f:
        json.dump(log_data, f)
    Logger.success("Evaluation logs saved to logs/evaluation_logs.json")

    # 3. Generate evaluation plots
    try:
        plot_evaluation(log_file="logs/evaluation_logs.json", output_dir="plots")
    except Exception as e:
        Logger.warning(f"Plotting failed: {e}")

    # 4. Print summary table
    Logger.section("Evaluation Summary")
    
    # Clearance time comparison
    time_diff = madrl_clearance - fixed_clearance
    time_pct = (time_diff / fixed_clearance * 100) if fixed_clearance > 0 else 0
    print(f"\nClearance Time:")
    print(f"  Fixed-Time: {fixed_clearance:.1f}s")
    print(f"  CF-MADRL:   {madrl_clearance:.1f}s ({time_diff:+.1f}s, {time_pct:+.1f}%)")
    
    if time_diff > 0:
        print(f"\nWARNING: CF-MADRL takes {time_diff:.1f}s LONGER despite better metrics!")
        print(f"   -> Possible causes:")
        print(f"      - Holding green too long when lanes are empty")
        print(f"      - Overly cautious duration choices (check duration analysis above)")
        print(f"      - Not switching fast enough in low-traffic periods")
    
    # Per-junction congestion metrics
    print(f"\nCongestion Metrics (Average Reward):")
    print(f"{'Junction ID':<35} | {'MADRL':>10} | {'Fixed':>10} | {'Improvement':>10}")
    print("-" * 80)

    for aid in junction_ids:
        r_madrl = np.mean(madrl_time_metrics[aid]["rewards"])
        r_fixed = np.mean(fixed_time_metrics[aid]["rewards"])

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
