import sys
import os
import torch

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from inference_deployment.pi_deploy.model import PPOPolicy


def show_summary():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.normpath(
        os.path.join(script_dir, "..", "pi_deploy", "model.pt")
    )

    # Defaults for 6 lanes, 4 phases, 6 durations
    obs_dim = 13
    act_dim = 24

    # Get dimensions from model file if it exists
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, weights_only=True)
            obs_dim = state_dict["network.0.weight"].shape[1]
            act_dim = state_dict["network.4.weight"].shape[0]
            print(
                f"Detected from {os.path.basename(model_path)}: Obs={obs_dim}, Act={act_dim}"
            )
        except Exception:
            print("Using default dimensions.")

    # Instantiate Model
    model = PPOPolicy(obs_dim, act_dim)

    print("\n" + "=" * 50)
    print("      PyTorch Model Architecture Summary")
    print("=" * 50)
    print(f"Model Class: {model.__class__.__name__}")
    print("-" * 50)

    print(f"{'Layer (type)':<25} {'Output Shape':<15} {'Param #'}")
    print("-" * 50)

    total_params = 0
    x = torch.zeros(1, obs_dim)  # Dummy input for shape tracking

    for name, layer in model.network.named_children():
        layer_type = layer.__class__.__name__
        x = layer(x)
        out_shape = str(list(x.shape[1:]))

        params = sum(p.numel() for p in layer.parameters())
        total_params += params
        print(f"{name + ' (' + layer_type + ')':<25} {out_shape:<15} {params:,}")

    print("-" * 50)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_params:,}")
    print("Non-trainable params: 0")
    print("-" * 50)

    params_mb = total_params * 4 / (1024**2)
    print(f"Params size (MB): {params_mb:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    show_summary()
