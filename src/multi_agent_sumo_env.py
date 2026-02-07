import numpy as np
import sumolib
import traci
import time
import os
import json

from gymnasium import spaces
from tools.utils import SilenceStdout, PrevTrafficState, RunningNorm, GlobalMetrics
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

        # GUI or Headless SUMO (use evaluation settings if in eval mode)
        if self.evaluation_mode:
            use_gui = self.config.get("evaluation", {}).get("gui", False)
            self.step_delay = self.config.get("evaluation", {}).get("step_delay", 0)
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

        # Max Steps Logic
        self.max_steps = self.config.get("training", {}).get(
            "local_steps_per_round", 1000
        )

        # To store previous traffic state per agent
        self.prev_state = PrevTrafficState()

        # RunningNorm for each agent
        self.queue_norms = {
            aid: RunningNorm(shape=self.max_lanes) for aid in self.junction_ids
        }
        self.wait_norms = {
            aid: RunningNorm(shape=self.max_lanes) for aid in self.junction_ids
        }

        # Duration Configuration
        duration_cfg = self.config.get("traffic", {}).get("durations", {})
        if not isinstance(duration_cfg, dict):
            duration_cfg = {}

        d_min = duration_cfg.get("min", 10)
        d_max = duration_cfg.get("max", 60)
        d_step = duration_cfg.get("step", 10)
        self.durations = list(range(d_min, d_max + 1, d_step))
        self.num_durations = len(self.durations)





        # NOTE - Track Current Phase for state
        self.current_phases = {aid: 0 for aid in self.junction_ids}
        
        
        # NOTE - Define Observation Space
        # Max_lanes(Queue) + Max_lanes(Wait) + Phase index
        obs_dim = (self.max_lanes * 2) + 1
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

        # Get initial observations
        observations = self._get_observations()
        infos = {aid: {} for aid in self.junction_ids}

        # Initialize previous traffic state (before any action)
        for aid in self.junction_ids:
            lanes = self.junction_metadata[aid]["lanes"]
            total_q = sum(
                self.conn.lane.getLastStepHaltingNumber(lane) for lane in lanes
            )
            total_w = sum(self.conn.lane.getWaitingTime(lane) for lane in lanes)
            self.prev_state.set(aid, total_q, total_w)

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

            # 3. Apply yellow phase.
            for _ in range(self.yellow_time):
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
            
            
            

        # Step 5: NOTE - Run Simulation
        max_duration = max(a["duration"] for a in agent_actions.values())
        rewards = {aid: 0.0 for aid in self.junction_ids}
        infos = {aid: {} for aid in self.junction_ids}

        for _ in range(max_duration):
            if self.conn.simulation.getMinExpectedNumber() <= 0:
                break
            self.conn.simulationStep()
            self.steps_counter += 1




        # Step 6: Compute reward
        for aid in self.junction_ids:
            lanes = self.junction_metadata[aid]["lanes"]

            total_q = sum(self.conn.lane.getLastStepHaltingNumber(ln) for ln in lanes)
            total_w = sum(self.conn.lane.getWaitingTime(ln) for ln in lanes)

            # Average negative penalty per lane (stable magnitude)
            # Dividing by len(lanes) makes reward invariant to junction size
            
            
            
            # NOTE - Rewards for each agent
            rewards[aid] = -1.0 * (total_q + total_w) / len(lanes)




            # Update previous state for next step
            self.prev_state.set(aid, total_q, total_w)

            # Track info for logging
            infos[aid]["step_queue"] = total_q
            infos[aid]["step_wait"] = total_w

            # Update global metrics for real-time logging
            GlobalMetrics.update(aid, rewards[aid], total_q, total_w)

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
    def _get_observations(self) -> Dict[str, np.ndarray]:
        observations = {}
        for agent_id in self.junction_ids:
            metadata = self.junction_metadata[agent_id]
            lanes = metadata["lanes"]
            
            
            
            
            
            
            # NOTE - Reading Traffic values from sumo for state calculation.
            queues = [self.conn.lane.getLastStepHaltingNumber(ln) for ln in lanes]
            waits = [self.conn.lane.getWaitingTime(ln) for ln in lanes]




            # Pad to max_lanes BEFORE normalization so shapes match RunningNorm(max_lanes)
            q_padded = np.array(queues + [0.0] * (self.max_lanes - len(queues)))
            w_padded = np.array(waits + [0.0] * (self.max_lanes - len(waits)))

            # Normalize using the padded arrays
            queues_norm = self.queue_norms[agent_id].normalize(q_padded)
            waits_norm = self.wait_norms[agent_id].normalize(w_padded)

            # Update stats
            self.queue_norms[agent_id].update(q_padded)
            self.wait_norms[agent_id].update(w_padded)

            # Clip values to ensure they stay within bounds [-10, 10]
            queues_norm = np.clip(queues_norm, -10.0, 10.0)
            waits_norm = np.clip(waits_norm, -10.0, 10.0)

            phase = self.conn.trafficlight.getPhase(agent_id)
            # print(f"Agent {agent_id} State | Raw Queues: {queues} | Raw Waits: {waits} | Current Phase: {phase}")
            phase_norm = phase / self.max_phases






            # NOTE - Normalized observation vector
            observations[agent_id] = np.array(
                list(queues_norm) + list(waits_norm) + [phase_norm],
                dtype=np.float32,
            )

            # So structure is:
            # Max_lanes(Queue) + Max_lanes(Wait) + Phase index
            # [ lane1_q, lane2_q, ..., q_pad, lane1_w, lane2_w, ..., W_PAD, current_phase ]

        return observations


    
    
    # Save normalization statistics for later
    def save_norm_stats(self, path: str):
        """Save RunningNorm stats for all agents."""
        stats = {}
        for aid in self.junction_ids:
            stats[aid] = {
                "queue": self.queue_norms[aid].get_state(),
                "wait": self.wait_norms[aid].get_state(),
            }

        # Convert numpy arrays to lists for JSON serialization
        def default_serializer(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(stats, f, default=default_serializer, indent=2)
        print(f"Normalization stats saved to {path}")
        
        
    
    
    # Use saved normalization to normalize sate vectors
    def load_norm_stats(self, path: str):
        """Load RunningNorm stats for all agents."""
        if not os.path.exists(path):
            return

        with open(path, "r") as f:
            stats = json.load(f)

        for aid, s in stats.items():
            if aid in self.queue_norms:
                self.queue_norms[aid].set_state(s["queue"])
            if aid in self.wait_norms:
                self.wait_norms[aid].set_state(s["wait"])
        print(f"Normalization stats loaded from {path}")



    # Close the environment
    def close(self):
        """Cleanly close the SUMO simulation."""

        if self.sim_active:
            try:
                self.conn.close()
            except Exception:
                pass
            self.sim_active = False
