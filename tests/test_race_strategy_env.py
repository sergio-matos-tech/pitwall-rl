"""Unit and edge-case tests for RaceStrategyEnv (RaceStrategy-v0)."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import src.envs  # noqa: F401 - Registers RaceStrategy-v0
from src.envs.race_strategy_env import (
    RaceAction,
    RaceConfig,
    RaceStrategyEnv,
    TireCompound,
)


def test_gymnasium_api_compliance() -> None:
    """Verify RaceStrategy-v0 passes Gymnasium's check_env."""
    env = gym.make("RaceStrategy-v0")
    check_env(env.unwrapped)
    env.close()


def test_spaces_and_seed_reproducibility() -> None:
    """Verify observation/action spaces and bit-exact seed determinism."""
    env_a = RaceStrategyEnv()
    env_b = RaceStrategyEnv()

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)

    assert obs_a.shape == (7,)
    assert obs_a.dtype == np.float32
    assert env_a.observation_space.contains(obs_a)
    assert env_a.action_space.n == 6
    assert np.array_equal(obs_a, obs_b)

    actions = [RaceAction.PUSH, RaceAction.BALANCED, RaceAction.CONSERVE] * 4
    rewards_a, rewards_b = [], []
    for action in actions:
        oa, ra, _, _, _ = env_a.step(action)
        ob, rb, _, _, _ = env_b.step(action)
        rewards_a.append(ra)
        rewards_b.append(rb)
        assert np.allclose(oa, ob)

    assert np.allclose(rewards_a, rewards_b)

    # Different seed should diverge stochastically
    env_c = RaceStrategyEnv()
    env_c.reset(seed=999)
    rewards_c = [env_c.step(action)[1] for action in actions]
    assert not np.allclose(rewards_a, rewards_c)


def test_invalid_action_raises_value_error() -> None:
    """Verify passing an out-of-range action raises ValueError."""
    env = RaceStrategyEnv()
    env.reset(seed=42)
    with pytest.raises(ValueError, match="Invalid action"):
        env.step(99)


def test_pit_stop_resets_wear_and_tracks_compounds() -> None:
    """Verify pitting resets tire wear, switches compound, and logs pit stop."""
    env = RaceStrategyEnv()
    env.reset(seed=7, options={"starting_compound": TireCompound.MEDIUM})

    # Run a few laps to build wear
    for _ in range(5):
        env.step(RaceAction.PUSH)

    obs, _, terminated, truncated, info = env.step(RaceAction.PIT_SOFT)
    assert not terminated and not truncated
    assert info["is_pit_stop"] is True
    assert info["tire_wear"] == 0.0
    assert info["current_compound"] == "SOFT"
    assert info["pit_stops_count"] == 1
    assert set(info["compounds_used"]) == {"MEDIUM", "SOFT"}
    assert info["valid_strategy"] is True
    assert obs[1] == 0.0  # tire_wear
    assert obs[2] == float(TireCompound.SOFT.value)
    assert obs[4] == 1.0  # pit_stops_count
    assert obs[6] == 2.0  # unique_compounds_used


def test_tire_puncture_causes_early_dnf_and_penalty() -> None:
    """Verify running worn Soft tires triggers puncture DNF before lap 50."""
    env = RaceStrategyEnv()
    env.reset(seed=42, options={"starting_compound": TireCompound.SOFT})

    terminated = False
    last_info = {}
    last_reward = 0.0
    laps = 0

    while not terminated and laps < env.config.total_laps:
        _, last_reward, terminated, _, last_info = env.step(RaceAction.PUSH)
        laps += 1

    assert terminated is True
    assert laps < env.config.total_laps
    assert last_info["is_puncture"] is True
    expected_reward = last_info["lap_reward"] - env.config.puncture_reward_penalty
    assert last_reward == pytest.approx(expected_reward)


def test_fia_two_compound_rule_terminal_rewards() -> None:
    """Verify +20 finish bonus vs -100 DSQ penalty at race completion."""
    cfg = RaceConfig()

    # 1) Single-compound race (Hard + Conserve survives 50 laps without pitting) -> DSQ
    env_illegal = RaceStrategyEnv(config=cfg)
    env_illegal.reset(seed=10, options={"starting_compound": TireCompound.HARD})

    final_reward_illegal = 0.0
    info_illegal = {}
    for _ in range(cfg.total_laps):
        _, final_reward_illegal, terminated, _, info_illegal = env_illegal.step(
            RaceAction.CONSERVE
        )

    assert terminated is True
    assert info_illegal["is_puncture"] is False
    assert info_illegal["valid_strategy"] is False
    assert final_reward_illegal == pytest.approx(
        info_illegal["lap_reward"] - cfg.illegal_strategy_penalty
    )

    # 2) Two-compound race (Medium -> Pit Hard on lap 25) -> Finish bonus
    env_legal = RaceStrategyEnv(config=cfg)
    env_legal.reset(seed=10, options={"starting_compound": TireCompound.MEDIUM})

    final_reward_legal = 0.0
    info_legal = {}
    for lap in range(1, cfg.total_laps + 1):
        action = RaceAction.PIT_HARD if lap == 25 else RaceAction.BALANCED
        _, final_reward_legal, terminated, _, info_legal = env_legal.step(action)

    assert terminated is True
    assert info_legal["is_puncture"] is False
    assert info_legal["valid_strategy"] is True
    assert final_reward_legal == pytest.approx(
        info_legal["lap_reward"] + cfg.finish_bonus
    )


def test_ansi_render_returns_telemetry_string() -> None:
    """Verify ansi render mode formats lap telemetry cleanly."""
    env = RaceStrategyEnv(render_mode="ansi")
    env.reset(seed=1)
    env.step(RaceAction.BALANCED)
    rendered = env.render()
    assert isinstance(rendered, str)
    assert "Lap 01/50" in rendered
    assert "MEDIUM" in rendered
