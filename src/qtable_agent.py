"""
Q-Table agent for multi-agent traffic signal control.

State: discretized queue counts per lane (4 bins) + current phase bin.
Action: integer index into (green_phase × duration) combinations.
"""

import os
import pickle
import numpy as np


class QTableAgent:
    """
    Tabular Q-Learning agent for a single junction.

    Parameters
    ----------
    junction_id   : str
    n_actions     : int   — total number of discrete actions
    lr            : float — learning rate (alpha)
    gamma         : float — discount factor
    epsilon       : float — initial exploration rate
    epsilon_min   : float — minimum exploration rate
    epsilon_decay : float — multiplicative decay per step
    n_bins        : int   — number of discretization bins
    bin_edges     : list  — boundaries for bins
    """

    def __init__(
        self,
        junction_id: str,
        n_actions: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9995,
        n_bins: int = 4,
        bin_edges: list = [-10.0, -0.5, 0.5, 2.0, 10.0],
    ):
        self.junction_id = junction_id
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins
        self.bin_edges = bin_edges

        # Q-table: dict mapping state_key → np.ndarray of shape (n_actions,)
        self.q_table: dict = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _discretize(self, value: float) -> int:
        """Map a normalized float to a bin index [0, self.n_bins-1]."""
        for i in range(len(self.bin_edges) - 1):
            if value <= self.bin_edges[i + 1]:
                return i
        return self.n_bins - 1

    def obs_to_state_key(self, obs: np.ndarray) -> tuple:
        """
        Convert a continuous observation vector to a discrete hashable state key.

        obs layout: [queue_norm × max_lanes, wait_norm × max_lanes, phase_norm]
        We discretize queue values + waiting times + phase.
        """
        n = len(obs)
        # Structure is: Max_lanes(Queue) + Max_lanes(Wait) + Phase index
        max_lanes = (n - 1) // 2

        queues = obs[:max_lanes]
        waits = obs[max_lanes : 2 * max_lanes]
        phase_norm = obs[-1]

        # Discretize everything
        q_bins = tuple(self._discretize(q) for q in queues)
        w_bins = tuple(self._discretize(w) for w in waits)
        phase_bin = int(round(phase_norm * 10))  # phase_norm → integer bucket

        return q_bins + w_bins + (phase_bin,)

    def _get_q(self, state_key: tuple) -> np.ndarray:
        """Return Q-values for a state, initializing to zeros if unseen."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[state_key]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        state_key = self.obs_to_state_key(obs)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self._get_q(state_key)))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """Standard Q-learning update (Bellman equation)."""
        s = self.obs_to_state_key(obs)
        s_next = self.obs_to_state_key(next_obs)

        q_current = self._get_q(s)[action]
        q_next = 0.0 if done else np.max(self._get_q(s_next))
        td_target = reward + self.gamma * q_next
        self._get_q(s)[action] += self.lr * (td_target - q_current)

    def save(self, path: str):
        """Persist Q-table and hyperparameters to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "n_actions": self.n_actions,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Load Q-table from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.n_actions = data.get("n_actions", self.n_actions)
