"""Regression and smoke tests for the Desafio 2 DQN training & evaluation pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_dqn import (
    evaluate_policy_suite,
    plot_race_telemetry_profile,
    plot_training_curves,
    train_dqn,
)
from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.envs.race_strategy_env import RaceStrategyEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "dqn_race_strategy.pt"


@pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(),
    reason="Pre-trained Desafio 2 checkpoint not found.",
)
def test_saved_checkpoint_beats_heuristic_across_20_seeds() -> None:
    """Verify the saved Desafio 2 model maintains 100% FIA compliance and beats baseline."""
    agent = DQNAgent(config=DQNConfig(), seed=42, device="cpu")
    meta = agent.load(CHECKPOINT_PATH)
    assert "episode" in meta

    eval_env = RaceStrategyEnv()
    eval_seeds = [42 + i for i in range(20)]

    heuristic_eval = evaluate_policy_suite(eval_env, "heuristic", eval_seeds)
    dqn_eval = evaluate_policy_suite(eval_env, "dqn", eval_seeds, agent=agent)

    assert dqn_eval["fia_compliance_pct"] == 100.0
    assert dqn_eval["dnf_rate_pct"] == 0.0
    assert dqn_eval["mean_reward"] > heuristic_eval["mean_reward"]
    assert dqn_eval["mean_time"] < heuristic_eval["mean_time"]


def test_short_training_and_plotting_smoke(tmp_path: Path) -> None:
    """Run a brief 50-episode training + plotting cycle to guard against pipeline regressions."""
    model_path = tmp_path / "dqn_smoke.pt"
    fig1_path = tmp_path / "training_curves.png"
    fig2_path = tmp_path / "telemetry_profile.png"

    agent, history, evaluations = train_dqn(
        num_episodes=50,
        seed=123,
        model_save_path=model_path,
    )

    assert len(history) == 50
    assert model_path.exists() and model_path.stat().st_size > 0
    assert set(evaluations.keys()) == {"random", "heuristic", "dqn"}

    plot_training_curves(history, evaluations, fig1_path)
    summary = plot_race_telemetry_profile(RaceStrategyEnv(), agent, fig2_path, seed=42)

    assert fig1_path.exists() and fig1_path.stat().st_size > 0
    assert fig2_path.exists() and fig2_path.stat().st_size > 0
    assert "dqn_seed42" in summary
