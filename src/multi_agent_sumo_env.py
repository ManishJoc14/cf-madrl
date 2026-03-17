import numpy as np
import sumolib
import traci
import time
import os
import json

from gymnasium import spaces
from tools.utils import SilenceStdout, GlobalMetrics
from typing import Dict, Any, Optional
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentSumoEnv(MultiAgentEnv):
    """
    Native RLlib Multi-Agent SUMO Traffic Environment.
    All agents control different junctions in a single shared SUMO simulation.
    """

    # Initializes the multi agent simulation environment
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__()

        # Agents = Junctions
        self.config = config_dict["config"]
        self.junction_ids = config_dict["junction_ids"]
        self._agent_ids = set(self.junction_ids)

        # Check if we're in evaluation mode (override GUI and delay settings)
        self.evaluation_mode = config_dict.get("evaluation_mode", False)

        # Lane & Phase Limits
        self.max_lanes = config_dict["max_lanes"]
        self.max_phases = config_dict["max_phases"]

        # SUMO Configuration
        self.sumo_config = self.config["sumo"]["config_file"]
        self.step_length = self.config["sumo"]["step_length"]

        # Traffic Light Timings
        self.yellow_time = self.config.get("traffic", {}).get("yellow_time", 3)
        self.green_time = self.config.get("traffic", {}).get("green_time", 10)

        # Reward Stabilization Params
        self.reward_wait_weight = self.config["rl"].get("reward_wait_weight", 0.1)
        self.reward_floor = self.config["rl"].get("reward_floor", -200.0)

        # GUI or Headless SUMO (use evaluation settings if in eval mode)
        if self.evaluation_mode:
            use_gui = self.config.get("evaluation", {}).get("gui", False)
            self.step_delay = config_dict.get(
                "step_delay", self.config.get("evaluation", {}).get("step_delay", 0)
            )
        else:
            use_gui = self.config["sumo"]["gui"]
            self.step_delay = 0
        if use_gui:
            try:
                self.sumo_binary = sumolib.checkBinary("sumo-gui")
            except Exception:
                print(
                    "Warning: sumo-gui not found or not supported. Falling back to sumo CLI."
                )
                self.sumo_binary = sumolib.checkBinary("sumo")
        else:
            self.sumo_binary = sumolib.checkBinary("sumo")

        # Connection State
        self.conn = None
        self.sim_active = False
        self.connection_label = f"ma_sumo_{np.random.randint(999999)}"

        # Environment memory
        self.junction_metadata = {}
        self.steps_counter = 0
        self.outgoing_lanes_by_junction = {}
        self.incoming_lane_sources_by_junction = {}
        self.source_target_outgoing_lanes = {}
        self.source_outgoing_lanes_all = {}
        self.latest_junction_metrics = {
            aid: {"queue": 0.0, "wait": 0.0} for aid in self.junction_ids
        }

        # Max Steps Logic
        self.max_steps = self.config.get("training", {}).get(
            "local_steps_per_round", 1000
        )

        # Static scaling divisors (replaces RunningNorm to prevent state drift)
        self.queue_scale = 15.0  # Normalized 1.0 = 20 vehicles halting
        self.wait_scale = 300.0  # Normalized 1.0 = 100 seconds total wait

        # Duration Configuration
        duration_cfg = self.config.get("traffic", {}).get(
            "durations", [10, 20, 30, 40, 50, 60]
        )
        if isinstance(duration_cfg, list):
            self.durations = duration_cfg
        else:
            # Fallback for old dict format
            d_min = duration_cfg.get("min", 10)
            d_max = duration_cfg.get("max", 60)
            d_step = duration_cfg.get("step", 10)
            self.durations = list(range(d_min, d_max + 1, d_step))
        self.num_durations = len(self.durations)

        # NOTE - Track Current Phase for state
        self.current_phases = {aid: 0 for aid in self.junction_ids}

        # NOTE - Define Observation Space
        # Max_lanes(Queue) + Max_lanes(Wait) + Max_lanes(SharedQueue)
        # + Max_lanes(SharedWait) + Phase index
        obs_dim = (self.max_lanes * 4) + 1
        single_obs = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # NOTE - Define Action Space
        total_actions = self.max_phases * self.num_durations
        single_act = spaces.Discrete(total_actions)

        # Final Assignment to RLlib
        self.observation_space = spaces.Dict(
            {aid: single_obs for aid in self.junction_ids}
        )
        self.action_space = spaces.Dict({aid: single_act for aid in self.junction_ids})

        # Attempt to load normalization stats if they exist
        stats_path = os.path.join(
            self.config["system"].get("log_dir", "logs"), "norm_stats.json"
        )
        if os.path.exists(stats_path):
            self.load_norm_stats(stats_path)

    # Resets the environment
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment and return initial observations for all agents."""

        # Seeding
        if seed is not None:
            np.random.seed(seed)

        # Reset step counter
        self.steps_counter = 0

        # Close existing SUMO connection (if any)
        if self.sim_active:
            try:
                self.conn.close()
            except Exception:
                pass

        # Build SUMO start command
        cmd = [
            self.sumo_binary,
            "-c",
            self.sumo_config,
            "--step-length",
            str(self.step_length),
            "--quit-on-end",
        ]

        # GUI support and delay
        if self.evaluation_mode:
            use_gui = self.config.get("evaluation", {}).get("gui", False)
            if use_gui:
                cmd.extend(["--start"])
                # Add SUMO's native delay parameter (in milliseconds)
                if self.step_delay > 0:
                    cmd.extend(["--delay", str(int(self.step_delay))])
        else:
            if self.config["sumo"]["gui"]:
                cmd.extend(["--start"])
            else:
                # NOTE - Fast training flags (headless only)
                # Suppress all internal SUMO logging and gridlock resolution
                cmd.extend(
                    [
                        "--no-step-log",
                        "--no-warnings",
                        "--duration-log.disable",
                        "--time-to-teleport",
                        "-1",
                        "--no-internal-links",  # skip internal lane geometry (faster)
                    ]
                )

        # Robust connection retry loop
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # numRetries=20 tells TraCI to wait longer for the socket to open
                with SilenceStdout():
                    traci.start(cmd, label=self.connection_label, numRetries=20)
                    time.sleep(1)
                    self.conn = traci.getConnection(self.connection_label)
                break
            except Exception as e:
                # Catch ALL startup errors (library path issues, headless crashes, etc.)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    try:
                        traci.close()
                    except Exception:
                        pass
                else:
                    raise e

        # Mark simulation as active
        self.sim_active = True

        # Discover junction metadata
        for j_id in self.junction_ids:
            # Ensure MADRL program is active (separate from fixed-time baseline)
            try:
                self.conn.trafficlight.setProgram(j_id, "madrl")
            except Exception:
                pass
            lanes = sorted(list(set(self.conn.trafficlight.getControlledLanes(j_id))))
            logics = self.conn.trafficlight.getCompleteRedYellowGreenDefinition(j_id)
            current_program_id = self.conn.trafficlight.getProgram(j_id)

            active_logic = next(
                (logic for logic in logics if logic.programID == current_program_id),
                logics[0],
            )

            # Find green phases index only
            green_phases = []
            for i, p in enumerate(active_logic.phases):
                if "G" in p.state or "g" in p.state:
                    green_phases.append(i)

            # Store metadata
            self.junction_metadata[j_id] = {
                "lanes": lanes,
                "green_phases": green_phases,
                "num_green": len(green_phases),
                "logic": active_logic,
            }

            # Initialize from first green phase
            start_p = green_phases[0] if green_phases else 0
            self.conn.trafficlight.setPhase(j_id, start_p)
            self.current_phases[j_id] = start_p

        # Build outgoing-lane map for sharing traffic metadata
        incoming_lane_to_junction = {}
        for j_id in self.junction_ids:
            for ln in self.junction_metadata[j_id]["lanes"]:
                incoming_lane_to_junction[ln] = j_id

        incoming_lane_sources = {}
        for j_id in self.junction_ids:
            outgoing = set()
            outgoing_all = set()
            per_target = {}
            try:
                controlled_links = self.conn.trafficlight.getControlledLinks(j_id)
            except Exception:
                controlled_links = []
            for link_group in controlled_links:
                for link in link_group:
                    if not link or len(link) < 2:
                        continue
                    to_lane = link[1]
                    if to_lane:
                        outgoing_all.add(to_lane)
                        outgoing.add(to_lane)
                        target = incoming_lane_to_junction.get(to_lane)
                        if target:
                            per_target.setdefault(target, set()).add(to_lane)
            self.outgoing_lanes_by_junction[j_id] = sorted(outgoing)
            self.source_outgoing_lanes_all[j_id] = sorted(outgoing_all)
            self.source_target_outgoing_lanes[j_id] = {
                tgt: sorted(list(lns)) for tgt, lns in per_target.items()
            }
            for ln in outgoing:
                incoming_lane_sources.setdefault(ln, set()).add(j_id)

        self.incoming_lane_sources_by_junction = {}
        for j_id in self.junction_ids:
            lanes = self.junction_metadata[j_id]["lanes"]
            self.incoming_lane_sources_by_junction[j_id] = [
                sorted([s for s in incoming_lane_sources.get(ln, []) if s != j_id])
                for ln in lanes
            ]

        self.latest_junction_metrics = {
            aid: {"queue": 0.0, "wait": 0.0} for aid in self.junction_ids
        }

        # Get initial observations
        observations = self._get_observations()
        infos = {aid: {} for aid in self.junction_ids}

        # Return
        return observations, infos

    # Applies actions of the agents to the environment
    def step(self, action_dict: Dict[str, int]):
        """Execute actions for all agents simultaneously."""

        # Safety check
        if not self.sim_active:
            # Return Empty observations, rewards, and a done flag (__all__) saying "episode over".
            return {}, {}, {"__all__": True}, {"__all__": True}, {}

        agent_actions = {}
        yellow_required = False

        # 0. Pre-populate to avoid KeyErrors
        for aid in self.junction_ids:
            # we pre-fill with current green phase and default duration.
            agent_actions[aid] = {
                "green": self.current_phases[aid],
                "duration": self.durations[0],
            }

        # 1. Decode actions and identify transitions
        for aid, action in action_dict.items():
            if aid not in self.junction_metadata:
                continue

            meta = self.junction_metadata[aid]

            num_g = meta["num_green"]

            # Revised Action Decoding:
            # action = (green_idx_local * num_durations) + dur_idx
            # This ensures even duration distribution across self.durations (e.g., [10, 20, ..., 60])

            # NOTE - Action decoding
            dur_idx = action % self.num_durations
            green_idx_local = (action // self.num_durations) % num_g

            duration_val = self.durations[min(dur_idx, self.num_durations - 1)]
            target_green_idx = meta["green_phases"][green_idx_local]

            # Update agent actions
            agent_actions[aid] = {"green": target_green_idx, "duration": duration_val}
            # print(f"Agent {aid} Action: Green Phase {target_green_idx}, Duration {duration_val}s")

            # If the agent wants a different green phase set yellow required
            if target_green_idx != self.current_phases[aid]:
                yellow_required = True
                # print(f"Agent {aid} Transition: Switching from phase {self.current_phases[aid]} to {target_green_idx} (Yellow Triggered)")

        # 2. Set Yellow Phases if needed
        if yellow_required:
            for aid in self.junction_ids:
                meta = self.junction_metadata[aid]
                curr_idx = self.current_phases[aid]
                target_idx = agent_actions[aid]["green"]
                if curr_idx != target_idx:
                    # SUMO defines phases in order: [green, yellow, red...]
                    y_idx = (curr_idx + 1) % len(meta["logic"].phases)
                    self.conn.trafficlight.setPhase(aid, y_idx)

            # 3. Apply yellow phase (scaled by step_length)
            num_yellow_steps = max(1, int(self.yellow_time / self.step_length))
            for _ in range(num_yellow_steps):
                self.conn.simulationStep()
                self.steps_counter += 1

        # 4. Set Green Phases
        for aid in self.junction_ids:
            target_idx = agent_actions[aid]["green"]
            # SUMO now switches the light for this junction to the agent's chosen green phase.

            # NOTE - Change phase in sumo
            self.conn.trafficlight.setPhase(aid, target_idx)
            # Update internal state
            self.current_phases[aid] = target_idx

        # Step 5: Run Simulation for calculated duration
        max_duration = max(a["duration"] for a in agent_actions.values())
        # num_sim_steps = duration / step_length (e.g. 30s / 10s = 3 steps)
        num_sim_steps = max(1, int(max_duration / self.step_length))

        rewards = {aid: 0.0 for aid in self.junction_ids}
        infos = {
            aid: {"step_queue": 0.0, "step_wait": 0.0} for aid in self.junction_ids
        }
        step_counts = {aid: 0 for aid in self.junction_ids}

        for _ in range(num_sim_steps):
            if self.conn.simulation.getMinExpectedNumber() <= 0:
                break
            self.conn.simulationStep()
            self.steps_counter += 1

            # Accumulate rewards and track metrics at EVERY simulation step
            for aid in self.junction_ids:
                lanes = self.junction_metadata[aid]["lanes"]
                try:
                    total_q = sum(
                        self.conn.lane.getLastStepHaltingNumber(ln) for ln in lanes
                    )
                    total_w = sum(self.conn.lane.getWaitingTime(ln) for ln in lanes)
                except (
                    traci.exceptions.FatalTraCIError,
                    traci.exceptions.TraCIException,
                ):
                    # Connection lost - mark simulation as inactive and return
                    self.sim_active = False
                    return {}, {}, {"__all__": True}, {"__all__": True}, {}

                # Penalize halting and waiting (Weighted)
                # total_w is cumulative and can explode, so we weight it down.
                step_reward = -(total_q + self.reward_wait_weight * total_w) / len(
                    lanes
                )
                rewards[aid] += step_reward

                # Accumulate for logs
                infos[aid]["step_queue"] = infos[aid].get("step_queue", 0) + total_q
                infos[aid]["step_wait"] = infos[aid].get("step_wait", 0) + total_w
                step_counts[aid] += 1

        # Step 6: Finalize rewards and info (Average over duration)
        for aid in self.junction_ids:
            if step_counts[aid] > 0:
                rewards[aid] /= step_counts[aid]

                # Apply Reward Floor to prevent extreme spikes
                rewards[aid] = max(self.reward_floor, rewards[aid])

                infos[aid]["step_queue"] /= step_counts[aid]
                infos[aid]["step_wait"] /= step_counts[aid]

            self.latest_junction_metrics[aid] = {
                "queue": float(infos[aid]["step_queue"]),
                "wait": float(infos[aid]["step_wait"]),
            }

            # Update global metrics for real-time logging
            GlobalMetrics.update(
                aid, rewards[aid], infos[aid]["step_queue"], infos[aid]["step_wait"]
            )

        # Step 7: Observations & Done Flags
        observations = self._get_observations()

        sim_done = self.conn.simulation.getMinExpectedNumber() <= 0
        limit_reached = self.steps_counter >= self.max_steps

        is_done = sim_done or limit_reached
        terminateds = {aid: is_done for aid in self.junction_ids}
        truncateds = {aid: False for aid in self.junction_ids}
        terminateds["__all__"] = is_done
        truncateds["__all__"] = False

        return observations, rewards, terminateds, truncateds, infos

    # Calculates current state observation from environment
    def _compute_turn_ratios(self) -> Dict[str, Dict[str, float]]:
        ratios = {}
        for src, all_out_lanes in self.source_outgoing_lanes_all.items():
            total = 0.0
            lane_counts = {}
            for ln in all_out_lanes:
                try:
                    cnt = self.conn.lane.getLastStepVehicleNumber(ln)
                except Exception:
                    cnt = 0.0
                lane_counts[ln] = float(cnt)
                total += float(cnt)

            targets = self.source_target_outgoing_lanes.get(src, {})
            if total > 0.0:
                ratios[src] = {
                    tgt: sum(lane_counts.get(ln, 0.0) for ln in lns) / total
                    for tgt, lns in targets.items()
                }
            else:
                num_targets = max(len(targets), 1)
                ratios[src] = {
                    tgt: 1.0 / num_targets for tgt in targets.keys()
                }
        return ratios

    def _get_observations(self) -> Dict[str, np.ndarray]:
        observations = {}
        turn_ratios = self._compute_turn_ratios()
        for agent_id in self.junction_ids:
            metadata = self.junction_metadata[agent_id]
            lanes = metadata["lanes"]

            # NOTE - Reading Traffic values from sumo for state calculation.
            queues = [self.conn.lane.getLastStepHaltingNumber(ln) for ln in lanes]
            waits = [self.conn.lane.getWaitingTime(ln) for ln in lanes]

            shared_sources = self.incoming_lane_sources_by_junction.get(agent_id, [])
            shared_queues = []
            shared_waits = []
            for idx, ln in enumerate(lanes):
                if idx < len(shared_sources) and shared_sources[idx]:
                    weighted_q = 0.0
                    weighted_w = 0.0
                    for src in shared_sources[idx]:
                        metrics = self.latest_junction_metrics.get(
                            src, {"queue": 0.0, "wait": 0.0}
                        )
                        w = turn_ratios.get(src, {}).get(agent_id, 0.0)
                        weighted_q += metrics["queue"] * w
                        weighted_w += metrics["wait"] * w
                    shared_queues.append(float(weighted_q))
                    shared_waits.append(float(weighted_w))
                else:
                    shared_queues.append(0.0)
                    shared_waits.append(0.0)

            # Pad to max_lanes BEFORE normalization so shapes match RunningNorm(max_lanes)
            q_padded = np.array(queues + [0.0] * (self.max_lanes - len(queues)))
            w_padded = np.array(waits + [0.0] * (self.max_lanes - len(waits)))
            sq_padded = np.array(
                shared_queues + [0.0] * (self.max_lanes - len(shared_queues))
            )
            sw_padded = np.array(
                shared_waits + [0.0] * (self.max_lanes - len(shared_waits))
            )

            # Static Scaling (semantics stay consistent across rounds)
            queues_norm = q_padded / self.queue_scale
            waits_norm = w_padded / self.wait_scale
            shared_queues_norm = sq_padded / self.queue_scale
            shared_waits_norm = sw_padded / self.wait_scale

            # Clip values to ensure they stay within bounds [-10, 10]
            queues_norm = np.clip(queues_norm, -10.0, 10.0)
            waits_norm = np.clip(waits_norm, -10.0, 10.0)
            shared_queues_norm = np.clip(shared_queues_norm, -10.0, 10.0)
            shared_waits_norm = np.clip(shared_waits_norm, -10.0, 10.0)

            phase = self.conn.trafficlight.getPhase(agent_id)
            # print(f"Agent {agent_id} State | Raw Queues: {queues} | Raw Waits: {waits} | Current Phase: {phase}")
            phase_norm = phase / self.max_phases

            # NOTE - Normalized observation vector
            observations[agent_id] = np.array(
                list(queues_norm)
                + list(waits_norm)
                + list(shared_queues_norm)
                + list(shared_waits_norm)
                + [phase_norm],
                dtype=np.float32,
            )

            # So structure is:
            # Max_lanes(Queue) + Max_lanes(Wait) + Max_lanes(SharedQueue)
            # + Max_lanes(SharedWait) + Phase index
            # [ lane1_q, ..., q_pad, lane1_w, ..., w_pad,
            #   lane1_sq, ..., sq_pad, lane1_sw, ..., sw_pad, current_phase ]

        return observations

    # Save scaling factors (replaces RunningNorm stats)
    def save_norm_stats(self, path: str):
        """Save scaling factors."""
        stats = {"queue_scale": self.queue_scale, "wait_scale": self.wait_scale}

        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    # Logic for static scaling is hardcoded, but we keep the method for consistency
    def load_norm_stats(self, path: str):
        """Stub for loading normalization if needed in future."""
        pass

    # Close the environment
    def close(self):
        """Cleanly close the SUMO simulation."""

        if self.sim_active:
            try:
                self.conn.close()
            except Exception:
                pass
            self.sim_active = False
