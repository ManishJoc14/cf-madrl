import torch
import torch.nn as nn


class PPOPolicy(nn.Module):
    """
    PyTorch implementation of the trained PPO policy network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=[128, 64], activation=nn.ReLU):
        super(PPOPolicy, self).__init__()
        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, act_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=-1).item()
