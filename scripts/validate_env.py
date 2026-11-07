"""Validation and Diagnostic Rollout Script for RaceStrategyEnv (Gymnasium v1.0+).

This script performs comprehensive validation:
1. Gymnasium API compliance via `gymnasium.utils.env_checker.check_env`.
2. Seed reproducibility verification (deterministic PRNG verification).
3. Random agent rollout baseline.
4. Strategic heuristic agent rollout (Medium -> Hard 1-stop strategy).
5. State space boundary and dtype integrity checks.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

# Ensure src is in python path
import src.envs
from src.envs.race_strategy_env import (
    PaceMode,
    RaceAction,
    RaceConfig,
    RaceStrategyEnv,
    TireCompound,
)


def run_compliance_check() -> None:
    """Run Farama Gymnasium standard environment compliance test."""
    print("=" * 70)
    print("1. RUNNING GYMNASIUM API SPECIFICATION COMPLIANCE CHECK")
    print("=" * 70)

    env = gym.make("RaceStrategy-v0")
    check_env(env.unwrapped)
    print("[PASS] Gymnasium check_env passed with 0 errors.\n")


def run_space_verification() -> None:
    """Verify bounds, shapes, and types for action and observation spaces."""
    print("=" * 70)
    print("2. VERIFYING SPACES & DTYPES")
    print("=" * 70)

    env = gym.make("RaceStrategy-v0")
    obs, info = env.reset(seed=100)

    # Observation space checks
    assert isinstance(obs, np.ndarray), f"Obs must be np.ndarray, got {type(obs)}"
    assert obs.dtype == np.float32, f"Obs dtype must be float32, got {obs.dtype}"
    assert obs.shape == (7,), f"Obs shape must be (7,), got {obs.shape}"
    assert env.observation_space.contains(obs), "Initial observation out of bounds!"

    # Action space checks
    assert env.action_space.n == 6, f"Expected 6 actions, got {env.action_space.n}"

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Initial obs vector: {np.round(obs, 4)}")
    print("[PASS] Space shapes, bounds, and dtypes strictly conform.\n")


def run_reproducibility_check() -> None:
    """Verify that identical seeds produce bit-exact identical trajectories."""
    print("=" * 70)
    print("3. VERIFYING SEED DETERMINISM & REPRODUCIBILITY")
    print("=" * 70)

    seed = 42
    fixed_actions = [RaceAction.PUSH, RaceAction.BALANCED, RaceAction.CONSERVE] * 5

    def rollout_fixed(s: int) -> list[float]:
        env = gym.make("RaceStrategy-v0")
        obs, _ = env.reset(seed=s)
        rewards = []
        for a in fixed_actions:
            obs, r, term, trunc, _ = env.step(a)
            rewards.append(r)
            if term or trunc:
                break
        return rewards

    run_a = rollout_fixed(seed)
    run_b = rollout_fixed(seed)
    run_different = rollout_fixed(seed + 999)

    assert np.allclose(run_a, run_b), "Runs with identical seeds must produce identical rewards!"
    assert not np.allclose(run_a, run_different), "Runs with different seeds should vary stochastically."

    print(f"Seed {seed} Run A rewards (first 3): {[round(x, 4) for x in run_a[:3]]}")
    print(f"Seed {seed} Run B rewards (first 3): {[round(x, 4) for x in run_b[:3]]}")
    print("[PASS] Environment PRNG is fully reproducible.\n")


def run_random_rollout() -> None:
    """Execute one full episode with a random stochastic agent."""
    print("=" * 70)
    print("4. RUNNING RANDOM AGENT ROLLOUT")
    print("=" * 70)

    env = gym.make("RaceStrategy-v0")
    obs, info = env.reset(seed=2026)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"Episode completed in {steps} laps.")
    print(f"Total Cumulative Reward: {total_reward:.2f}")
    print(f"Final Cumulative Time:   {info['cumulative_time']:.2f} s")
    print(f"Pit Stops Count:         {info['pit_stops_count']}")
    print(f"Compounds Used:          {info['compounds_used']}")
    print(f"Valid Strategy (>=2 cmp):{info['valid_strategy']}")
    print("[PASS] Random rollout finished successfully.\n")


def run_heuristic_strategy_rollout() -> None:
    """Execute a realistic 1-stop pit strategy (Medium -> Hard at lap 25)."""
    print("=" * 70)
    print("5. RUNNING HEURISTIC EXPERT STRATEGY ROLLOUT (1-STOP M -> H)")
    print("=" * 70)

    env = gym.make("RaceStrategy-v0")
    obs, info = env.reset(seed=42, options={"starting_compound": TireCompound.MEDIUM})

    total_reward = 0.0
    done = False
    print(f"{'Lap':>4} | {'Action':<10} | {'Tire':<7} | {'Wear':>5} | {'Fuel':>6} | {'Lap Time':>8} | {'Total Time':>10}")
    print("-" * 65)

    lap = 1
    while not done:
        # Heuristic Strategy:
        # - Laps 1-24: Balanced pace on Mediums
        # - Lap 25: Pit for Hard tires
        # - Laps 26-44: Balanced pace on fresh Hards
        # - Laps 45-50: Push pace on light fuel to the finish line
        if lap == 25:
            action = RaceAction.PIT_HARD
        elif lap >= 45:
            action = RaceAction.PUSH
        else:
            action = RaceAction.BALANCED

        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Print telemetry for key laps
        if lap in (1, 5, 15, 24, 25, 26, 35, 45, 49, 50):
            act_name = RaceAction(action).name
            print(
                f"{lap:4d} | {act_name:<10} | {step_info['current_compound']:<7} | "
                f"{step_info['tire_wear']*100:4.1f}% | {step_info['fuel_kg']:5.1f}k | "
                f"{step_info['lap_time']:7.2f}s | {step_info['cumulative_time']:9.2f}s"
            )
        lap += 1

    print("-" * 65)
    print(f"Strategic 1-Stop Cumulative Race Time: {step_info['cumulative_time']:.2f} s")
    print(f"Total Cumulative Reward:              {total_reward:.2f}")
    print(f"Unique Compounds Used:                {step_info['compounds_used']}")
    print(f"FIA Strategy Compliance:              {step_info['valid_strategy']}")
    print("[PASS] Expert heuristic strategy executed flawlessly.\n")


def main() -> None:
    """Run all validation suites."""
    print("=" * 70)
    print("STARTING COMPREHENSIVE VALIDATION: RaceStrategy-v0")
    print("=" * 70)

    try:
        run_compliance_check()
        run_space_verification()
        run_reproducibility_check()
        run_random_rollout()
        run_heuristic_strategy_rollout()
        print("=" * 70)
        print("ALL TESTS & BENCHMARKS PASSED SUCCESSFULLY!")
        print("=" * 70)
    except Exception as exc:
        print(f"\n[FAIL] Validation encountered an error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
