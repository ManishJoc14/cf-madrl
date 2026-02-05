import os
import time
import torch
import numpy as np
from model import PPOPolicy
from inference_utils import RunningNorm


class TrafficInference:
    """
    Traffic control inference engine designed to run on edge devices like Raspberry Pi.
    Uses a standalone PyTorch model exported from the training environment.
    """

    def __init__(
        self,
        model_filename="model.pt",
        max_lanes=6,
        max_phases=4,
        num_green_phases=2,
        green_phases_indices=None,
    ):
        self.max_lanes = max_lanes
        self.max_phases = max_phases
        self.num_green_phases = (
            num_green_phases  # Actual number of green phases for this junction
        )
        # List of actual SUMO phase indices that are green ([0, 2] for J5)
        self.green_phases_indices = green_phases_indices or list(
            range(num_green_phases)
        )
        self.durations = [10, 20, 30, 40, 50, 60]
        self.num_durations = len(self.durations)
        self.last_phase = None

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, model_filename)

        self.queue_norm = RunningNorm(shape=max_lanes)
        self.wait_norm = RunningNorm(shape=max_lanes)
        self.policy = self._load_model()

        # Load norm stats if available
        stats_path = os.path.join(script_dir, "norm_stats.json")
        if os.path.exists(stats_path):
            self._load_norm_stats(stats_path)

    def _load_norm_stats(self, path):
        import json

        try:
            with open(path, "r") as f:
                stats = json.load(f)
            # Find J5 or just use first agent in stats if only J5 exists
            agent_id = next(iter(stats.keys()))  # Default to first key
            if "J5" in stats:
                agent_id = "J5"

            self.queue_norm.set_state(stats[agent_id]["queue"])
            self.wait_norm.set_state(stats[agent_id]["wait"])
            print(f"Normalization stats loaded for agent {agent_id} from {path}")
        except Exception as e:
            print(f"Warning: Failed to load norm stats: {e}")

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        try:
            state_dict = torch.load(self.model_path, weights_only=True)
            obs_dim = state_dict["network.0.weight"].shape[1]
            act_dim = state_dict["network.4.weight"].shape[0]

            policy = PPOPolicy(obs_dim, act_dim)
            policy.load_state_dict(state_dict)
            policy.eval()
            print(f"Model Loaded | Obs: {obs_dim}, Actions: {act_dim}")
            return policy
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")

    def get_action(self, queues, waits, current_phase):
        """
        Returns (target_phase, duration, yellow_required) based on traffic state.
        """
        # Pad inputs to match trained model shape (6 lanes)
        q_padded = np.array(list(queues) + [0.0] * (self.max_lanes - len(queues)))
        w_padded = np.array(list(waits) + [0.0] * (self.max_lanes - len(waits)))

        q_norm = self.queue_norm.normalize(q_padded)
        w_norm = self.wait_norm.normalize(w_padded)

        self.queue_norm.update(q_padded)
        self.wait_norm.update(w_padded)

        state = np.concatenate(
            [
                np.clip(q_norm, -10.0, 10.0),
                np.clip(w_norm, -10.0, 10.0),
                [current_phase / self.max_phases],
            ]
        ).astype(np.float32)

        action_id = self.policy.predict(state)

        # Action Decoding: action = (green_idx_local * num_durations) + dur_idx
        dur_idx = action_id % self.num_durations
        green_idx_local = (action_id // self.num_durations) % self.num_green_phases

        duration_val = self.durations[min(dur_idx, self.num_durations - 1)]

        # Map local green index to actual SUMO phase index
        target_phase = self.green_phases_indices[green_idx_local]

        # Debugging prints to explain the model's choices
        print(f"DEBUG | Raw State (Norm): {np.round(state[:8], 2)}")
        print(
            f"DEBUG | Action ID: {action_id} -> Local Green Index: {green_idx_local}, Dur Index: {dur_idx}"
        )
        print(f"DEBUG | Decoded -> Phase: {target_phase}, Duration: {duration_val}s")

        yellow_required = (
            self.last_phase is not None and target_phase != self.last_phase
        )
        self.last_phase = target_phase

        return target_phase, duration_val, yellow_required


def run_inference():
    """Example loop for testing J5 deployment."""
    print("Starting Inference Engine (J5)...")

    # Configuration for J5
    MAX_TRAINED_LANES = 6
    J5_REAL_LANES = 4
    MAX_PHASES = 4
    NUM_GREEN_PHASES = 2  # J5 has 2 green phases
    GREEN_PHASE_INDICES = [0, 2]  # Actual SUMO indices for green phases in J5
    YELLOW_TIME = 3

    try:
        engine = TrafficInference(
            max_lanes=MAX_TRAINED_LANES,
            max_phases=MAX_PHASES,
            num_green_phases=NUM_GREEN_PHASES,
            green_phases_indices=GREEN_PHASE_INDICES,
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    # Inputs must be in lexicographical order (alphabetical) for the model to work.
    # For J5, these were: [-E14_0, -E9_0, E11_0, E4_0]
    # Predefined test scenarios with intuitive traffic patterns
    # J5 Phase Logic: Phase 2 → Lanes [0,3] GREEN | Phase 0 → Lanes [1,2] GREEN
    # Each scenario: (name, [lane0, lane1, lane2, lane3], [waits], expected_insight)
    test_scenarios = [
        # Scenario 1: Heavy congestion on lanes 0 & 3 → should select phase 2 to clear them
        ("Heavy lanes 0&3", [9, 1, 1, 8], [28, 3, 4, 25], "Expect phase 2 (clear 0&3)"),
        # Scenario 2: Heavy congestion on lanes 1 & 2 → should select phase 0 to clear them
        ("Heavy lanes 1&2", [1, 8, 9, 1], [2, 27, 30, 3], "Expect phase 0 (clear 1&2)"),
        # Scenario 3: Only lane 0 congested → should select phase 2
        ("Lane 0 heavy", [10, 0, 0, 0], [30, 0, 0, 0], "Expect phase 2"),
        # Scenario 4: Only lane 3 congested → should select phase 2 (lane 3 gets green in phase 2)
        ("Lane 3 heavy", [0, 0, 0, 10], [0, 0, 0, 30], "Expect phase 2"),
        # Scenario 5: Only lane 1 congested → should select phase 0
        ("Lane 1 heavy", [0, 10, 0, 0], [0, 30, 0, 0], "Expect phase 0"),
        # Scenario 6: Only lane 2 congested → should select phase 0
        ("Lane 2 heavy", [0, 0, 10, 0], [0, 0, 30, 0], "Expect phase 0"),
        # Scenario 7: Balanced traffic
        ("Balanced traffic", [5, 5, 5, 5], [15, 15, 15, 15], "Either phase OK"),
        # Scenario 8: All clear
        ("No congestion", [0, 0, 0, 0], [0, 0, 0, 0], "Maintain current"),
    ]

    current_phase = 0
    step = 0
    try:
        while True:
            # Cycle through test scenarios
            scenario_idx = step % len(test_scenarios)
            scenario_name, sim_queues, sim_waits, insight = test_scenarios[scenario_idx]

            print(f"\n{'=' * 60}")
            print(f"Step {step:03} | Scenario: {scenario_name}")
            print(f"         | Queues: {sim_queues} | Waits: {sim_waits}")
            print(f"         | Active Phase: {current_phase} | {insight}")
            print(f"{'=' * 60}")

            target_phase, duration, yellow_needed = engine.get_action(
                sim_queues, sim_waits, current_phase
            )

            if yellow_needed:
                print(
                    f"         | [YELLOW] Switching {current_phase} -> {target_phase} ({YELLOW_TIME}s)"
                )
                time.sleep(YELLOW_TIME)

            print(f"         | [GREEN] Phase {target_phase} for {duration}s")
            current_phase = target_phase
            time.sleep(duration)

            print("-" * 30)
            step += 1
    except KeyboardInterrupt:
        print("\nDeployment stopped.")


if __name__ == "__main__":
    run_inference()
