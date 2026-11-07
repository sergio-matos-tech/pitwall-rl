"""Deep Q-Network (DQN) Agent for RaceStrategy-v0.

Implements a clean, defensive, Double-DQN agent with Experience Replay,
Polyak soft target updates, Huber Loss (SmoothL1Loss), and gradient clipping
to handle non-linear tire degradation cliffs and terminal regulatory penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass(frozen=True)
class DQNConfig:
    """Immutable hyperparameter configuration for the DQN agent."""

    state_dim: int = 7
    action_dim: int = 6
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 50_000
    target_update_tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_episodes: int = 1400
    max_grad_norm: float = 1.0
    warmup_steps: int = 640
    updates_per_step: int = 2
    pit_explore_prob: float = 0.018


class QNetwork(nn.Module):
    """Multi-Layer Perceptron (MLP) Q-value approximator.

    Architecture:
        Linear(7, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, 6)
    """

    state_offset: torch.Tensor
    state_scale: torch.Tensor

    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        """Initialize the MLP layers and feature normalization buffers."""
        super().__init__()
        self.register_buffer(
            "state_offset",
            torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32),
        )
        self.register_buffer(
            "state_scale",
            torch.tensor([1.0, 1.0, 2.0, 1.0, 5.0, 30.0, 2.0], dtype=torch.float32),
        )
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for all actions given a batch of states."""
        normalized = (state - self.state_offset) / self.state_scale
        return self.net(normalized)


class ReplayBuffer:
    """Vectorized circular Experience Replay Buffer with pre-allocated NumPy arrays."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device,
    ) -> None:
        """Initialize pre-allocated memory arrays for transitions."""
        if capacity <= 0:
            raise ValueError(f"Buffer capacity must be positive, got {capacity}.")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.device = device

        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self._ptr: int = 0
        self._size: int = 0

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self._size

    def push(
        self,
        state: npt.NDArray[np.float32],
        action: int,
        reward: float,
        next_state: npt.NDArray[np.float32],
        done: bool,
    ) -> None:
        """Store a single transition tuple (s, a, r, s', done) in the buffer."""
        self.states[self._ptr] = state
        self.actions[self._ptr, 0] = int(action)
        self.rewards[self._ptr, 0] = float(reward)
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr, 0] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a uniform random batch of transitions as PyTorch tensors."""
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample batch of size {batch_size} from buffer with size {self._size}."
            )

        if rng is not None:
            indices = rng.integers(0, self._size, size=batch_size)
        else:
            indices = np.random.randint(0, self._size, size=batch_size)

        states_t = torch.as_tensor(self.states[indices], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(self.actions[indices], dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            self.next_states[indices], dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t


class DQNAgent:
    """Deep Q-Network Agent with Double-DQN targets and Polyak soft updates."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        seed: int = 42,
        device: str | torch.device | None = None,
    ) -> None:
        """Initialize online/target networks, optimizer, and replay buffer."""
        self.config = config if config is not None else DQNConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.online_net = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.lr)
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            state_dim=self.config.state_dim,
            device=self.device,
        )

        self.epsilon: float = self.config.epsilon_start
        self.total_steps: int = 0

    def get_epsilon(self, episode: int) -> float:
        """Compute linearly decayed epsilon for the given episode index."""
        if episode >= self.config.epsilon_decay_episodes:
            return self.config.epsilon_end
        fraction = float(episode) / float(max(1, self.config.epsilon_decay_episodes))
        return float(
            self.config.epsilon_start
            + fraction * (self.config.epsilon_end - self.config.epsilon_start)
        )

    def update_epsilon(self, episode: int) -> float:
        """Update and return the agent's current exploration rate."""
        self.epsilon = self.get_epsilon(episode)
        return self.epsilon

    def select_action(
        self,
        state: npt.NDArray[np.float32],
        eval_mode: bool = False,
    ) -> int:
        """Select an action using epsilon-greedy (training) or greedy argmax (eval).

        During random exploration, pace actions {0, 1, 2} and pit-stop actions {3, 4, 5}
        are sampled with domain-aware probabilities so the agent explores full tire stints
        instead of pitting every second lap.
        """
        if not eval_mode and self.rng.random() < self.epsilon:
            if self.rng.random() < self.config.pit_explore_prob:
                return int(self.rng.integers(3, self.config.action_dim))
            return int(self.rng.integers(0, 3))

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def step(
        self,
        state: npt.NDArray[np.float32],
        action: int,
        reward: float,
        next_state: npt.NDArray[np.float32],
        done: bool,
    ) -> float | None:
        """Record transition in replay buffer and perform gradient update steps."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.memory) < max(self.config.batch_size, self.config.warmup_steps):
            return None

        loss_val = 0.0
        for _ in range(self.config.updates_per_step):
            loss_val = self._optimize_model()
            self._soft_update_target()
        return loss_val

    def _optimize_model(self) -> float:
        """Execute one Double-DQN mini-batch gradient descent step with Huber loss."""
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.config.batch_size, rng=self.rng
        )

        # Current Q(s, a) from online network
        current_q = self.online_net(states).gather(1, actions)

        # Double-DQN target: a* = argmax_a Q_online(s', a), evaluate with Q_target(s', a*)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.config.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=self.config.max_grad_norm
        )
        self.optimizer.step()

        return float(loss.item())

    def _soft_update_target(self) -> None:
        """Perform Polyak averaging update on the target network parameters."""
        tau = self.config.target_update_tau
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        """Save online/target model state dicts and configuration to disk."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "config": self.config,
            "metadata": metadata or {},
        }
        torch.save(payload, save_path)

    def load(self, path: str | Path) -> dict[str, Any]:
        """Load model weights from checkpoint path."""
        load_path = Path(path)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.online_net.eval()
        self.target_net.eval()
        return checkpoint.get("metadata", {})
