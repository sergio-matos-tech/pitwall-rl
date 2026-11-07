"""Custom Gymnasium Environments package."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.registration import register

from src.envs.race_strategy_env import (
    PaceMode,
    RaceAction,
    RaceConfig,
    RaceStrategyEnv,
    TireCompound,
    TireProperties,
)

# Register the environment with Gymnasium's global environment registry
register(
    id="RaceStrategy-v0",
    entry_point="src.envs.race_strategy_env:RaceStrategyEnv",
    max_episode_steps=60,
)

__all__ = [
    "RaceStrategyEnv",
    "RaceConfig",
    "TireCompound",
    "PaceMode",
    "RaceAction",
    "TireProperties",
]
