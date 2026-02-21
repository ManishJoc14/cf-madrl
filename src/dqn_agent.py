"""
DQN agent for multi-agent traffic signal control.

Architecture: 2-layer MLP (128 → 64 → n_actions) with replay buffer.
One independent DQN per junction (no parameter sharing).
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------------ #
# Neural network
# ------------------------------------------------------------------ #


class QNetwork(nn.Module):
    """Dynamically built MLP Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_layers: list = [256, 256]):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------ #
# Replay buffer
# ------------------------------------------------------------------ #


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------------ #
# DQN Agent
# ------------------------------------------------------------------ #


class DQNAgent:
    """
    Independent DQN agent for a single junction.

    Parameters
    ----------
    junction_id        : str
    obs_dim            : int   — observation vector length
    n_actions          : int   — number of discrete actions
    lr                 : float — Adam learning rate
    gamma              : float — discount factor
    epsilon            : float — initial exploration rate
    epsilon_min        : float — minimum exploration rate
    epsilon_decay      : float — multiplicative decay per step
    batch_size         : int   — replay batch size
    buffer_size        : int   — replay buffer capacity
    target_update_freq : int   — sync target net every N steps
    fcnet_hiddens      : list  — hidden layer sizes
    """

    def __init__(
        self,
        junction_id: str,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        buffer_size: int = 10_000,
        target_update_freq: int = 200,
        fcnet_hiddens: list = [256, 256],
    ):
        self.junction_id = junction_id
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self._target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network
        self.q_net = QNetwork(obs_dim, n_actions, fcnet_hiddens).to(self.device)
        # Target network (periodically synced)
        self.target_net = QNetwork(obs_dim, n_actions, fcnet_hiddens).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)

        self._steps = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_net(obs_t)
        return int(q_vals.argmax(dim=1).item())

    def store(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self) -> float:
        """
        Sample a batch and perform one gradient update.
        Returns the loss value (0.0 if buffer not ready).
        """
        if len(self.buffer) < self.batch_size:
            return 0.0

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        obs_t = torch.FloatTensor(obs).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_obs_t = torch.FloatTensor(next_obs).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values (using target network)
        with torch.no_grad():
            max_next_q = self.target_net(next_obs_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q * (1.0 - dones_t)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._steps += 1

        # Sync target network periodically
        if self._steps % self._target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        """Save model weights and epsilon to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon": self.epsilon,
                "steps": self._steps,
            },
            path,
        )

    def load(self, path: str):
        """Load model weights from disk."""
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self._steps = ckpt.get("steps", 0)
        self.q_net.eval()
