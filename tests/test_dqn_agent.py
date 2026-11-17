"""Unit tests for DQNAgent, QNetwork, and ReplayBuffer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.dqn_agent import DQNAgent, DQNConfig, QNetwork, ReplayBuffer


def test_replay_buffer_push_wraparound_and_sample() -> None:
    """Verify circular buffer overwrite behavior, error checks, and sample tensors."""
    buf = ReplayBuffer(capacity=5, state_dim=7, device=torch.device("cpu"))
    assert len(buf) == 0

    with pytest.raises(ValueError, match="Cannot sample batch"):
        buf.sample(batch_size=2)

    for i in range(8):
        s = np.full((7,), float(i), dtype=np.float32)
        ns = np.full((7,), float(i + 1), dtype=np.float32)
        buf.push(s, action=i % 6, reward=-float(i), next_state=ns, done=(i == 7))

    assert len(buf) == 5
    # Buffer capacity 5 after 8 pushes should retain indices 3..7
    assert set(buf.states[:, 0].tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}

    rng = np.random.default_rng(42)
    states, actions, rewards, next_states, dones = buf.sample(batch_size=4, rng=rng)
    assert states.shape == (4, 7) and states.dtype == torch.float32
    assert actions.shape == (4, 1) and actions.dtype == torch.int64
    assert rewards.shape == (4, 1) and rewards.dtype == torch.float32
    assert next_states.shape == (4, 7) and next_states.dtype == torch.float32
    assert dones.shape == (4, 1) and dones.dtype == torch.float32


def test_qnetwork_forward_shape_and_finiteness() -> None:
    """Verify QNetwork produces finite Q-values of shape (batch, action_dim)."""
    net = QNetwork(state_dim=7, action_dim=6, hidden_dim=32)
    batch = torch.randn(10, 7, dtype=torch.float32)
    q_vals = net(batch)
    assert q_vals.shape == (10, 6)
    assert torch.isfinite(q_vals).all()


def test_epsilon_decay_schedule() -> None:
    """Verify linear epsilon decay boundaries."""
    cfg = DQNConfig(epsilon_start=1.0, epsilon_end=0.02, epsilon_decay_episodes=100)
    agent = DQNAgent(config=cfg, seed=42, device="cpu")

    assert agent.get_epsilon(0) == pytest.approx(1.0)
    assert agent.get_epsilon(50) == pytest.approx(0.51)
    assert agent.get_epsilon(100) == pytest.approx(0.02)
    assert agent.get_epsilon(500) == pytest.approx(0.02)


def test_agent_warmup_optimization_and_polyak_update() -> None:
    """Verify warmup skips updates, then optimization returns finite loss and updates target."""
    cfg = DQNConfig(
        hidden_dim=32,
        batch_size=4,
        buffer_capacity=100,
        warmup_steps=8,
        updates_per_step=2,
        target_update_tau=0.1,
    )
    agent = DQNAgent(config=cfg, seed=42, device="cpu")
    rng = np.random.default_rng(0)

    initial_target_params = [p.detach().clone() for p in agent.target_net.parameters()]

    # First 7 steps: below warmup_steps=8 -> returns None, target unchanged
    for _ in range(7):
        s = rng.random(7, dtype=np.float32)
        ns = rng.random(7, dtype=np.float32)
        loss = agent.step(s, action=1, reward=-1.0, next_state=ns, done=False)
        assert loss is None

    for p0, p1 in zip(initial_target_params, agent.target_net.parameters()):
        assert torch.equal(p0, p1)

    # Step 8+: triggers gradient update + Polyak soft update
    s = rng.random(7, dtype=np.float32)
    ns = rng.random(7, dtype=np.float32)
    loss = agent.step(s, action=1, reward=-1.0, next_state=ns, done=False)
    assert isinstance(loss, float)
    assert np.isfinite(loss)

    diffs = [
        torch.max(torch.abs(p0 - p1)).item()
        for p0, p1 in zip(initial_target_params, agent.target_net.parameters())
    ]
    assert max(diffs) > 0.0


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """Verify checkpoint save/load restores metadata and deterministic greedy policy."""
    cfg = DQNConfig(hidden_dim=32)
    agent = DQNAgent(config=cfg, seed=123, device="cpu")

    probe_states = np.random.default_rng(7).random((10, 7), dtype=np.float32)
    actions_before = [agent.select_action(s, eval_mode=True) for s in probe_states]

    ckpt_path = tmp_path / "agent_ckpt.pt"
    agent.save(ckpt_path, metadata={"episode": 42, "tag": "unit-test"})
    assert ckpt_path.exists()

    restored = DQNAgent(config=cfg, seed=999, device="cpu")
    meta = restored.load(ckpt_path)

    assert meta == {"episode": 42, "tag": "unit-test"}
    actions_after = [restored.select_action(s, eval_mode=True) for s in probe_states]
    assert actions_before == actions_after
