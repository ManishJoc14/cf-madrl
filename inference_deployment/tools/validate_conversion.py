import sys
import os
import torch
import numpy as np

# Add project root to path to import project modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from agent_manager import AgentManager
from multi_agent_sumo_env import MultiAgentSumoEnv
from utils import load_config, scan_topology
from inference_deployment.pi_deploy.model import PPOPolicy


def validate():
    print("Starting Model Validation...")

    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    config_path = os.path.join(root_dir, "config.yaml")
    model_path = os.path.join(script_dir, "..", "pi_deploy", "model.pt")

    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        return

    # Load Original RLlib Model
    config = load_config(config_path)
    junction_ids = config["system"]["controlled_junctions"]
    max_lanes, max_phases = scan_topology(config)

    env_config = {
        "config": config,
        "junction_ids": junction_ids,
        "max_lanes": max_lanes,
        "max_phases": max_phases,
    }

    env = MultiAgentSumoEnv(env_config)
    agent_manager = AgentManager(env, config, junction_ids)
    agent_manager.load(os.path.join(root_dir, "saved_models", "universal_models"))

    target_agent = "J5" if "J5" in junction_ids else junction_ids[0]
    print(f"Validating against RLlib Policy for: {target_agent}")
    rllib_policy = agent_manager.algo.get_policy(target_agent)

    # Load Standalone Model
    state_dict = torch.load(model_path, weights_only=True)
    obs_dim = state_dict["network.0.weight"].shape[1]
    act_dim = state_dict["network.4.weight"].shape[0]

    standalone_policy = PPOPolicy(obs_dim, act_dim)
    standalone_policy.load_state_dict(state_dict)
    standalone_policy.eval()

    # Compare Outputs on random data
    print(f"Comparing outputs for obs_dim={obs_dim}...")

    for i in range(5):
        obs = np.random.uniform(-1, 1, size=(obs_dim,)).astype(np.float32)

        # RLlib prediction
        with torch.no_grad():
            rllib_obs_batch = torch.from_numpy(obs).unsqueeze(0)
            rllib_out, _ = rllib_policy.model({"obs": rllib_obs_batch})
            rllib_logits = rllib_out.numpy()[0]

        # Standalone prediction
        with torch.no_grad():
            standalone_logits = standalone_policy.forward(torch.from_numpy(obs)).numpy()

        # Check match
        max_diff = np.max(np.abs(rllib_logits - standalone_logits))

        if max_diff < 1e-5:
            print(f"  Test {i + 1}: MATCH (Max Diff: {max_diff:.2e})")
        else:
            print(f"  Test {i + 1}: MISMATCH! (Max Diff: {max_diff:.2e})")

    # Cleanup
    agent_manager.close()
    env.close()


if __name__ == "__main__":
    validate()
