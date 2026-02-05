import sys
import os
import torch

# Add project root to path to import project modules
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from agent_manager import AgentManager
from multi_agent_sumo_env import MultiAgentSumoEnv
from utils import load_config, scan_topology


def export_weights():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    config_path = os.path.join(root_dir, "config.yaml")

    # Load Configuration
    if not os.path.exists(config_path):
        print(f"Error: config.yaml not found at {config_path}")
        return

    config = load_config(config_path)
    junction_ids = config["system"]["controlled_junctions"]

    # Scan Topology for environment setup
    print("Scanning topology...")
    max_lanes, max_phases = scan_topology(config)

    # Initialize environment stack (RLlib) to load weights
    print("Initializing environment stack...")
    env_config = {
        "config": config,
        "junction_ids": junction_ids,
        "max_lanes": max_lanes,
        "max_phases": max_phases,
    }

    env = MultiAgentSumoEnv(env_config)
    agent_manager = AgentManager(env, config, junction_ids)

    # Load Checkpoint from universal_models folder
    checkpoint_dir = os.path.join(root_dir, "saved_models", "universal_models")
    if os.path.exists(checkpoint_dir) and any(os.scandir(checkpoint_dir)):
        print(f"Loading checkpoint from: {checkpoint_dir}")
        try:
            agent_manager.load(checkpoint_dir)
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return
    else:
        print("No checkpoint found!")
        return

    # Extract Weights from target agent (J5 preferred)
    print("Extracting weights...")
    ray_weights = agent_manager.get_weights()

    if "J5" in junction_ids:
        target_agent = "J5"
    else:
        target_agent = junction_ids[0]

    print(f"Exporting weights from agent: {target_agent}")
    weights_numpy = ray_weights[target_agent]

    # Map RLlib keys to simple PPOPolicy keys
    state_dict = {}

    def map_key(k):
        if "_hidden_layers.0." in k:
            return k.replace("_hidden_layers.0._model.0", "network.0")
        if "_hidden_layers.1." in k:
            return k.replace("_hidden_layers.1._model.0", "network.2")
        if "_logits." in k:
            return k.replace("_logits._model.0", "network.4")
        return None

    print("Converting keys...")
    for k, v in weights_numpy.items():
        new_k = map_key(k)
        if new_k:
            state_dict[new_k] = torch.tensor(v)
            print(f"  {k} -> {new_k} ({v.shape})")
        else:
            print(f"  Skipping key: {k}")

    # Save to pi_deploy/model.pt
    output_path = os.path.join(script_dir, "..", "pi_deploy", "model.pt")
    torch.save(state_dict, output_path)
    print(f"Weights saved to: {os.path.abspath(output_path)}")

    # Cleanup
    agent_manager.close()
    env.close()


if __name__ == "__main__":
    export_weights()
