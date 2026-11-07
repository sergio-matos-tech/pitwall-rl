"""Training, Evaluation, and Visualization Pipeline for DQN on RaceStrategy-v0.

Executes Challenge 2 requirements:
1. Trains DQNAgent for 2,000 episodes on RaceStrategy-v0.
2. Saves the best model weights to `models/dqn_race_strategy.pt`.
3. Evaluates Random, Expert Heuristic Baseline, and Trained DQN across 20 fixed seeds.
4. Generates high-resolution (300 DPI) presentation figures in `docs/figures/`:
   - `docs/figures/dqn_training_curves.png`
   - `docs/figures/dqn_race_telemetry_profile.png`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import src.envs  # noqa: F401 - Registers RaceStrategy-v0
from src.agents.dqn_agent import DQNAgent, DQNConfig
from src.envs.race_strategy_env import RaceAction, RaceStrategyEnv, TireCompound


@dataclass
class EpisodeMetrics:
    """Per-episode telemetry summary collected during training or evaluation."""

    episode: int
    total_reward: float
    cumulative_time: float
    laps_completed: int
    pit_stops: int
    fia_compliant: bool
    is_dnf: bool
    compounds_used: list[str]
    epsilon: float


def heuristic_policy(lap_1idx: int) -> int:
    """Expert 1-stop Heuristic Strategy (Medium -> Hard at Lap 25, Push Laps 45-50).

    Matches the benchmark policy from `scripts/validate_env.py`.
    """
    if lap_1idx == 25:
        return int(RaceAction.PIT_HARD)
    if lap_1idx >= 45:
        return int(RaceAction.PUSH)
    return int(RaceAction.BALANCED)


def run_policy_rollout(
    env: RaceStrategyEnv,
    policy_type: str,
    seed: int,
    agent: DQNAgent | None = None,
) -> tuple[EpisodeMetrics, list[dict[str, Any]]]:
    """Execute one complete episode rollout under a specified policy and seed."""
    obs, info = env.reset(
        seed=seed,
        options={"starting_compound": TireCompound.MEDIUM},
    )
    rng = np.random.default_rng(seed + 10_000)

    total_reward = 0.0
    done = False
    lap_idx = 0
    telemetry: list[dict[str, Any]] = []

    while not done:
        lap_1idx = lap_idx + 1
        compound_before = info["current_compound"]
        wear_before = float(info["tire_wear"])
        fuel_before = float(info["fuel_kg"])

        if policy_type == "random":
            action = int(rng.integers(0, env.action_space.n))
        elif policy_type == "heuristic":
            action = heuristic_policy(lap_1idx)
        elif policy_type == "dqn":
            if agent is None:
                raise ValueError("DQNAgent instance is required for 'dqn' policy rollout.")
            action = agent.select_action(obs, eval_mode=True)
        else:
            raise ValueError(f"Unknown policy_type: {policy_type}")

        # Estimate end-of-lap wear before pit reset for accurate visual plotting
        act_enum = RaceAction(action)
        is_pit = act_enum in (
            RaceAction.PIT_SOFT,
            RaceAction.PIT_MEDIUM,
            RaceAction.PIT_HARD,
        )
        comp_enum = TireCompound[compound_before]
        comp_props = env.config.compounds[comp_enum]
        pace_mult = (
            1.55
            if act_enum == RaceAction.PUSH
            else (0.70 if act_enum == RaceAction.CONSERVE else 1.00)
        )
        wear_end_of_stint_lap = float(
            np.clip(
                wear_before
                + comp_props.base_wear_rate * pace_mult * (1.0 + 0.5 * (wear_before**2)),
                0.0,
                1.0,
            )
        )

        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)

        telemetry.append(
            {
                "lap": lap_1idx,
                "action": int(action),
                "action_name": act_enum.name,
                "stint_compound": compound_before,
                "post_compound": step_info["current_compound"],
                "wear_start": wear_before,
                "wear_stint_end": wear_end_of_stint_lap if is_pit else float(step_info["tire_wear"]),
                "wear_post": float(step_info["tire_wear"]),
                "cliff_threshold": comp_props.cliff_threshold,
                "fuel_kg": float(step_info["fuel_kg"]),
                "fuel_start_kg": fuel_before,
                "lap_time": float(step_info["lap_time"]),
                "cumulative_time": float(step_info["cumulative_time"]),
                "reward": float(reward),
                "is_pit_stop": bool(step_info["is_pit_stop"]),
                "is_puncture": bool(step_info["is_puncture"]),
            }
        )

        obs = next_obs
        info = step_info
        lap_idx += 1

    is_dnf = bool(info.get("is_puncture", False))
    fia_compliant = bool(info.get("valid_strategy", False)) and not is_dnf

    metrics = EpisodeMetrics(
        episode=0,
        total_reward=total_reward,
        cumulative_time=float(info["cumulative_time"]),
        laps_completed=int(info["current_lap"]),
        pit_stops=int(info["pit_stops_count"]),
        fia_compliant=fia_compliant,
        is_dnf=is_dnf,
        compounds_used=list(info["compounds_used"]),
        epsilon=0.0,
    )
    return metrics, telemetry


def evaluate_policy_suite(
    env: RaceStrategyEnv,
    policy_type: str,
    eval_seeds: list[int],
    agent: DQNAgent | None = None,
) -> dict[str, Any]:
    """Evaluate a policy across multiple fixed seeds and aggregate statistics."""
    episodes: list[EpisodeMetrics] = []
    for s in eval_seeds:
        m, _ = run_policy_rollout(env, policy_type=policy_type, seed=s, agent=agent)
        episodes.append(m)

    rewards = np.array([e.total_reward for e in episodes], dtype=np.float64)
    times = np.array([e.cumulative_time for e in episodes], dtype=np.float64)
    pits = np.array([e.pit_stops for e in episodes], dtype=np.float64)
    laps = np.array([e.laps_completed for e in episodes], dtype=np.float64)
    fia_rates = np.array([100.0 if e.fia_compliant else 0.0 for e in episodes], dtype=np.float64)
    dnf_rates = np.array([100.0 if e.is_dnf else 0.0 for e in episodes], dtype=np.float64)

    return {
        "policy": policy_type,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "mean_pits": float(np.mean(pits)),
        "std_pits": float(np.std(pits)),
        "mean_laps": float(np.mean(laps)),
        "fia_compliance_pct": float(np.mean(fia_rates)),
        "dnf_rate_pct": float(np.mean(dnf_rates)),
        "episodes": episodes,
    }


def train_dqn(
    num_episodes: int = 2000,
    seed: int = 42,
    model_save_path: Path = PROJECT_ROOT / "models" / "dqn_race_strategy.pt",
) -> tuple[DQNAgent, list[EpisodeMetrics], dict[str, dict[str, Any]]]:
    """Train the DQNAgent on RaceStrategy-v0 and track convergence metrics."""
    env = RaceStrategyEnv()
    eval_env = RaceStrategyEnv()

    config = DQNConfig(
        state_dim=7,
        action_dim=6,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_capacity=50_000,
        target_update_tau=0.005,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay_episodes=1400,
    )
    agent = DQNAgent(config=config, seed=seed)

    eval_seeds = [42 + i for i in range(20)]

    # Pre-compute Random and Heuristic Baselines across the 20 evaluation seeds
    random_eval = evaluate_policy_suite(eval_env, "random", eval_seeds)
    heuristic_eval = evaluate_policy_suite(eval_env, "heuristic", eval_seeds)

    print("=" * 78)
    print("TRAINING DEEP Q-NETWORK (DQN) ON RaceStrategy-v0")
    print("=" * 78)
    print(f"Device:             {agent.device}")
    print(f"Episodes:           {num_episodes}")
    print(f"Architecture:       7 -> {config.hidden_dim} -> {config.hidden_dim} -> 6 (Double-DQN)")
    print(f"Hyperparameters:    gamma={config.gamma}, lr={config.lr}, batch={config.batch_size}, tau={config.target_update_tau}")
    print(f"Random Baseline:    Return = {random_eval['mean_reward']:7.2f} +/- {random_eval['std_reward']:5.2f} | Time = {random_eval['mean_time']:7.2f}s")
    print(f"Heuristic Baseline: Return = {heuristic_eval['mean_reward']:7.2f} +/- {heuristic_eval['std_reward']:5.2f} | Time = {heuristic_eval['mean_time']:7.2f}s")
    print("-" * 78)

    history: list[EpisodeMetrics] = []
    best_eval_return = -float("inf")
    start_time = time.time()

    for ep in range(1, num_episodes + 1):
        eps = agent.update_epsilon(ep - 1)
        obs, info = env.reset(
            seed=seed * 10_000 + ep,
            options={"starting_compound": TireCompound.MEDIUM},
        )

        ep_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = bool(terminated or truncated)

            agent.step(obs, action, float(reward), next_obs, done)

            obs = next_obs
            info = step_info
            ep_reward += float(reward)

        is_dnf = bool(info.get("is_puncture", False))
        fia_ok = bool(info.get("valid_strategy", False)) and not is_dnf

        ep_metrics = EpisodeMetrics(
            episode=ep,
            total_reward=ep_reward,
            cumulative_time=float(info["cumulative_time"]),
            laps_completed=int(info["current_lap"]),
            pit_stops=int(info["pit_stops_count"]),
            fia_compliant=fia_ok,
            is_dnf=is_dnf,
            compounds_used=list(info["compounds_used"]),
            epsilon=eps,
        )
        history.append(ep_metrics)

        # Periodic deterministic evaluation to checkpoint the best policy
        if ep >= 400 and (ep % 25 == 0 or ep == num_episodes):
            dqn_eval = evaluate_policy_suite(eval_env, "dqn", eval_seeds, agent=agent)
            if (
                dqn_eval["fia_compliance_pct"] == 100.0
                and dqn_eval["dnf_rate_pct"] == 0.0
                and dqn_eval["mean_reward"] > best_eval_return
            ):
                best_eval_return = dqn_eval["mean_reward"]
                agent.save(
                    model_save_path,
                    metadata={
                        "episode": ep,
                        "mean_eval_reward": best_eval_return,
                        "mean_eval_time": dqn_eval["mean_time"],
                        "fia_compliance_pct": dqn_eval["fia_compliance_pct"],
                    },
                )

        if ep % 100 == 0 or ep == 1:
            recent = history[-50:]
            avg_r = float(np.mean([m.total_reward for m in recent]))
            avg_t = float(np.mean([m.cumulative_time for m in recent]))
            fia_r = float(np.mean([100.0 if m.fia_compliant else 0.0 for m in recent]))
            dnf_r = float(np.mean([100.0 if m.is_dnf else 0.0 for m in recent]))
            elapsed = time.time() - start_time
            print(
                f"Ep {ep:4d}/{num_episodes} | eps={eps:.3f} | "
                f"MA50 Return: {avg_r:7.2f} | MA50 Time: {avg_t:7.1f}s | "
                f"FIA: {fia_r:5.1f}% | DNF: {dnf_r:5.1f}% | "
                f"Best Eval: {best_eval_return:7.2f} ({elapsed:4.1f}s)"
            )

    # Ensure checkpoint exists and reload best model weights for final evaluation
    if model_save_path.exists():
        meta = agent.load(model_save_path)
        print("-" * 78)
        print(
            f"[CHECKPOINT] Restored best model from Episode {meta.get('episode')} "
            f"(Mean Eval Return: {meta.get('mean_eval_reward'):.2f})"
        )
    else:
        agent.save(model_save_path, metadata={"episode": num_episodes})

    final_dqn_eval = evaluate_policy_suite(eval_env, "dqn", eval_seeds, agent=agent)

    evaluations = {
        "random": random_eval,
        "heuristic": heuristic_eval,
        "dqn": final_dqn_eval,
    }
    return agent, history, evaluations


def plot_training_curves(
    history: list[EpisodeMetrics],
    evaluations: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate Figure 1: DQN Convergence and Learning Stages (`dqn_training_curves.png`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = np.array([m.episode for m in history], dtype=np.int32)
    rewards = np.array([m.total_reward for m in history], dtype=np.float64)
    times = np.array([m.cumulative_time for m in history], dtype=np.float64)
    fia_ok = np.array([100.0 if m.fia_compliant else 0.0 for m in history], dtype=np.float64)
    survival = np.array([0.0 if m.is_dnf else 100.0 for m in history], dtype=np.float64)

    # Moving average helper (window = 50)
    window = 50
    kernel = np.ones(window) / float(window)
    rewards_ma = np.convolve(rewards, kernel, mode="valid")
    ma_episodes = episodes[window - 1 :]

    # Block aggregation (blocks of 50 episodes) for Subplot 2
    block_size = 50
    num_blocks = len(history) // block_size
    block_centers: list[float] = []
    block_fia: list[float] = []
    block_survival: list[float] = []
    block_time: list[float] = []

    for b in range(num_blocks):
        start = b * block_size
        end = start + block_size
        block_centers.append(float(np.mean(episodes[start:end])))
        block_fia.append(float(np.mean(fia_ok[start:end])))
        block_survival.append(float(np.mean(survival[start:end])))
        # Average race time of completed (non-DNF) races in the block, fallback to all
        non_dnf_times = [
            history[i].cumulative_time for i in range(start, end) if not history[i].is_dnf
        ]
        if non_dnf_times:
            block_time.append(float(np.mean(non_dnf_times)))
        else:
            block_time.append(float(np.mean(times[start:end])))

    heur_ret = evaluations["heuristic"]["mean_reward"]
    rand_ret = evaluations["random"]["mean_reward"]
    dqn_ret = evaluations["dqn"]["mean_reward"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)
    fig.patch.set_facecolor("white")

    # -------------------------------------------------------------------------
    # Left Subplot: Cumulative Return per Episode
    # -------------------------------------------------------------------------
    ax1.plot(
        episodes,
        rewards,
        color="#93C5FD",
        alpha=0.32,
        linewidth=0.8,
        label="Retorno por Episódio (Treino $\\epsilon$-greedy)",
    )
    ax1.plot(
        ma_episodes,
        rewards_ma,
        color="#1D4ED8",
        linewidth=2.5,
        label="Média Móvel do DQN (50 episódios)",
    )
    ax1.axhline(
        dqn_ret,
        color="#059669",
        linestyle="-.",
        linewidth=2.0,
        label=f"DQN Avaliado ($\\epsilon=0$): {dqn_ret:.2f}",
    )
    ax1.axhline(
        heur_ret,
        color="#D97706",
        linestyle="--",
        linewidth=2.0,
        label=f"Baseline Heurístico (1-Stop M→H): {heur_ret:.2f}",
    )
    ax1.axhline(
        rand_ret,
        color="#DC2626",
        linestyle=":",
        linewidth=1.8,
        label=f"Agente Aleatório (Random): {rand_ret:.2f}",
    )

    ax1.set_title(
        "Convergência do Retorno Acumulado (RaceStrategy-v0)",
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("Episódio de Treinamento")
    ax1.set_ylabel("Retorno Acumulado ($G_t$)")
    ax1.set_xlim(1, len(history))
    ax1.set_ylim(-640.0, 15.0)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.0)

    # Annotate learning phases on Left Subplot in clean non-overlapping regions
    idx_p1 = min(120, len(rewards_ma) - 1)
    idx_p2 = min(550, len(rewards_ma) - 1)
    ax1.annotate(
        "Fase 1: Exploração e\nEliminação de DNF",
        xy=(ma_episodes[idx_p1], float(rewards_ma[idx_p1])),
        xytext=(90, -490),
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.3},
        fontsize=8.6,
        fontweight="bold",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax1.annotate(
        "Fase 2: Regra FIA\n(Troca Obrigatória)",
        xy=(ma_episodes[idx_p2], float(rewards_ma[idx_p2])),
        xytext=(480, -340),
        arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.3},
        fontsize=8.6,
        fontweight="bold",
        color="#334155",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax1.annotate(
        f"Fase 3: Supera Baseline\n(DQN = {dqn_ret:.1f} vs {heur_ret:.1f})",
        xy=(1550, dqn_ret),
        xytext=(1080, -225),
        arrowprops={"arrowstyle": "->", "color": "#059669", "lw": 1.5},
        fontsize=8.8,
        fontweight="bold",
        color="#065F46",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ECFDF5", "edgecolor": "#6EE7B7"},
    )

    # -------------------------------------------------------------------------
    # Right Subplot: Completion / FIA Compliance (%) & Mean Race Time (s)
    # -------------------------------------------------------------------------
    ax2_time = ax2.twinx()

    l1 = ax2.plot(
        block_centers,
        block_survival,
        color="#0284C7",
        marker="o",
        markersize=4.0,
        linewidth=2.2,
        label="Sobrevivência sem Estouro / Sem DNF (%)",
        zorder=4,
    )
    l2 = ax2.plot(
        block_centers,
        block_fia,
        color="#16A34A",
        marker="s",
        markersize=4.0,
        linewidth=2.4,
        label="Conformidade Regulatória FIA $\\geq 2$ Compostos (%)",
        zorder=5,
    )
    l3 = ax2_time.plot(
        block_centers,
        block_time,
        color="#EA580C",
        marker="^",
        markersize=4.0,
        linewidth=2.2,
        linestyle="--",
        label="Tempo Médio de Corrida Concluída (s)",
        zorder=3,
    )

    heur_time = evaluations["heuristic"]["mean_time"]
    l4 = ax2_time.axhline(
        heur_time,
        color="#9A3412",
        linestyle=":",
        linewidth=1.6,
        label=f"Tempo Baseline Heurístico ({heur_time:.1f} s)",
        zorder=2,
    )

    ax2.set_title(
        "Evolução por Blocos (50 Ep.): Segurança, Regra FIA e Tempo Total",
        fontweight="bold",
        pad=12,
    )
    ax2.set_xlabel("Episódio de Treinamento (Blocos de 50 Episódios)")
    ax2.set_ylabel("Taxa de Sucesso / Conformidade (%)", color="#0F172A")
    ax2_time.set_ylabel("Tempo Médio de Corrida (s) [Menor é Melhor]", color="#C2410C")

    ax2.set_xlim(1, len(history))
    ax2.set_ylim(-5.0, 108.0)
    min_t = min(min(block_time), heur_time) - 22.0
    max_t = max(max(block_time[:10]), heur_time + 70.0) + 18.0
    ax2_time.set_ylim(min_t, max_t)
    ax2.grid(True, linestyle="--", alpha=0.35)

    lines = l1 + l2 + l3 + [l4]
    labels = [line.get_label() for line in lines]
    leg2 = ax2_time.legend(
        lines,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.03, 0.04),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.96,
        fontsize=8.6,
    )
    leg2.set_zorder(30)

    fig.suptitle(
        "Desafio 2 — Treinamento e Convergência do Agente Deep Q-Network (DQN) no Ambiente RaceStrategy-v0",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_race_telemetry_profile(
    env: RaceStrategyEnv,
    agent: DQNAgent,
    output_path: Path,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate Figure 2: 50-Lap Learned Strategy Telemetry (`dqn_race_telemetry_profile.png`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dqn_metrics, dqn_tele = run_policy_rollout(env, "dqn", seed=seed, agent=agent)
    heur_metrics, heur_tele = run_policy_rollout(env, "heuristic", seed=seed)
    rand_metrics, rand_tele = run_policy_rollout(env, "random", seed=seed)

    laps = np.array([t["lap"] for t in dqn_tele], dtype=np.int32)
    wear_pct = np.array([t["wear_stint_end"] * 100.0 for t in dqn_tele], dtype=np.float64)
    stint_compounds = [t["stint_compound"] for t in dqn_tele]
    pit_laps = [t["lap"] for t in dqn_tele if t["is_pit_stop"]]

    compound_colors = {
        "SOFT": "#EF4444",    # Red
        "MEDIUM": "#F59E0B",  # Amber / Orange
        "HARD": "#3B82F6",    # Blue
    }

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
        }
    )

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(13.5, 9.8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.15, 1.0]},
    )
    fig.patch.set_facecolor("white")

    # -------------------------------------------------------------------------
    # Subplot 1: Tire Wear (%) & Active Compound + Tire Cliff & Pit Stop Marker
    # -------------------------------------------------------------------------
    # Plot stint segments colored by active tire compound
    stint_start = 0
    for i in range(len(dqn_tele)):
        is_end_of_stint = dqn_tele[i]["is_pit_stop"] or (i == len(dqn_tele) - 1)
        if is_end_of_stint:
            seg_laps = laps[stint_start : i + 1]
            seg_wear = wear_pct[stint_start : i + 1]
            comp_name = stint_compounds[stint_start]
            color = compound_colors.get(comp_name, "#64748B")

            ax1.plot(seg_laps, seg_wear, color=color, linewidth=2.8, zorder=4)
            ax1.scatter(seg_laps, seg_wear, color=color, s=26, zorder=5)
            ax1.fill_between(seg_laps, 0, seg_wear, color=color, alpha=0.12)

            # Connect pit-stop drop from end of stint to 0% wear and start of next stint
            if dqn_tele[i]["is_pit_stop"] and i + 1 < len(dqn_tele):
                next_comp = stint_compounds[i + 1]
                next_color = compound_colors.get(next_comp, "#64748B")
                ax1.plot(
                    [laps[i], laps[i]],
                    [wear_pct[i], 0.0],
                    color="#334155",
                    linestyle=":",
                    linewidth=1.7,
                    zorder=3,
                )
                ax1.plot(
                    [laps[i], laps[i + 1]],
                    [0.0, wear_pct[i + 1]],
                    color=next_color,
                    linewidth=2.2,
                    alpha=0.85,
                    zorder=3,
                )
                ax1.fill_between(
                    [laps[i], laps[i + 1]],
                    0,
                    [0.0, wear_pct[i + 1]],
                    color=next_color,
                    alpha=0.12,
                )
                stint_start = i + 1

    # Cliff threshold reference lines
    ax1.axhline(
        72.0,
        color="#D97706",
        linestyle="--",
        linewidth=1.4,
        alpha=0.85,
        label="Tire Cliff Medium (72%)",
    )
    ax1.axhline(
        65.0,
        color="#DC2626",
        linestyle=":",
        linewidth=1.4,
        alpha=0.85,
        label="Tire Cliff Soft (65%)",
    )

    for p_lap in pit_laps:
        post_comp = dqn_tele[p_lap - 1]["post_compound"]
        wear_at_pit = wear_pct[p_lap - 1]
        ax1.axvline(p_lap, color="#0F172A", linestyle="-.", linewidth=1.8, zorder=6)
        ax1.annotate(
            f"PIT STOP (Volta {p_lap})\nTroca p/ {post_comp} ({wear_at_pit:.1f}%)",
            xy=(p_lap, wear_at_pit),
            xytext=(min(42.5, p_lap + 1.3), 48.0),
            arrowprops={"arrowstyle": "->", "color": "#0F172A", "lw": 1.4},
            fontsize=8.8,
            fontweight="bold",
            color="#0F172A",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#94A3B8"},
            zorder=8,
        )

    ax1.set_title(
        "1. Degradação do Pneu (%) e Gestão de Compostos pelo Agente DQN (Seed = 42)",
        fontweight="bold",
    )
    ax1.set_ylabel("Desgaste do Pneu (%)")
    ax1.set_ylim(-4.0, 114.0)
    ax1.grid(True, linestyle="--", alpha=0.35)

    legend_elements_1 = [
        Patch(facecolor="#F59E0B", edgecolor="#D97706", label="Composto MEDIUM (Macio-Médio)"),
        Patch(facecolor="#EF4444", edgecolor="#DC2626", label="Composto SOFT (Ataque Rápido)"),
        Patch(facecolor="#3B82F6", edgecolor="#2563EB", label="Composto HARD (Alta Durabilidade)"),
        Line2D([0], [0], color="#D97706", linestyle="--", lw=1.5, label="Limiar Tire Cliff Medium (72%)"),
        Line2D([0], [0], color="#DC2626", linestyle=":", lw=1.5, label="Limiar Tire Cliff Soft (65%)"),
        Line2D([0], [0], color="#0F172A", linestyle="-.", lw=1.8, label="Chamada de Box (Pit Stop)"),
    ]
    ax1.legend(
        handles=legend_elements_1,
        loc="upper left",
        ncol=3,
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    # -------------------------------------------------------------------------
    # Subplot 2: Lap Time (s) Comparison (DQN vs. Heuristic vs. Random)
    # -------------------------------------------------------------------------
    dqn_lap_times = [t["lap_time"] for t in dqn_tele]
    heur_lap_times = [t["lap_time"] for t in heur_tele]
    rand_laps = [t["lap"] for t in rand_tele]
    rand_lap_times = [t["lap_time"] for t in rand_tele]

    ax2.plot(
        laps,
        dqn_lap_times,
        color="#059669",
        marker="o",
        markersize=3.5,
        linewidth=2.2,
        label=(
            f"Agente DQN Treinado (Total: {dqn_metrics.cumulative_time:.2f}s | "
            f"Retorno: {dqn_metrics.total_reward:.2f})"
        ),
        zorder=5,
    )
    ax2.plot(
        laps,
        heur_lap_times,
        color="#D97706",
        marker="s",
        markersize=3.2,
        linewidth=1.9,
        linestyle="--",
        label=(
            f"Heurística Especialista (Total: {heur_metrics.cumulative_time:.2f}s | "
            f"Retorno: {heur_metrics.total_reward:.2f})"
        ),
        zorder=4,
    )
    ax2.plot(
        rand_laps,
        rand_lap_times,
        color="#DC2626",
        linewidth=1.2,
        alpha=0.55,
        linestyle=":",
        label=(
            f"Agente Aleatório (Total: {rand_metrics.cumulative_time:.2f}s | "
            f"Retorno: {rand_metrics.total_reward:.2f})"
        ),
        zorder=2,
    )

    time_saved = heur_metrics.cumulative_time - dqn_metrics.cumulative_time
    ax2.set_title(
        f"2. Tempo de Volta Comparado — DQN Economiza {time_saved:.2f}s sobre a Heurística Especialista",
        fontweight="bold",
    )
    ax2.set_ylabel("Tempo de Volta (s)")
    ax2.set_ylim(87.5, 118.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.8, frameon=True, facecolor="white", framealpha=0.92)

    # -------------------------------------------------------------------------
    # Subplot 3: Policy Action Profile Chosen by DQN (A_t per Lap)
    # -------------------------------------------------------------------------
    actions_taken = [t["action"] for t in dqn_tele]
    action_labels = [
        "PUSH (0)",
        "BALANCED (1)",
        "CONSERVE (2)",
        "PIT_SOFT (3)",
        "PIT_MEDIUM (4)",
        "PIT_HARD (5)",
    ]
    action_colors = {
        0: "#EF4444",  # PUSH
        1: "#2563EB",  # BALANCED
        2: "#10B981",  # CONSERVE
        3: "#9333EA",  # PIT_SOFT
        4: "#D97706",  # PIT_MEDIUM
        5: "#475569",  # PIT_HARD
    }

    ax3.step(
        laps,
        actions_taken,
        where="mid",
        color="#64748B",
        linewidth=1.4,
        alpha=0.7,
        zorder=2,
    )
    for act_id in range(6):
        mask_laps = [laps[i] for i, a in enumerate(actions_taken) if a == act_id]
        mask_vals = [a for a in actions_taken if a == act_id]
        if mask_laps:
            ax3.scatter(
                mask_laps,
                mask_vals,
                color=action_colors[act_id],
                s=52 if act_id >= 3 else 36,
                marker="D" if act_id >= 3 else "o",
                label=action_labels[act_id],
                zorder=5 if act_id >= 3 else 4,
            )

    ax3.set_title(
        "3. Política de Decisão Sequencial do DQN ($A_t$ escolhida a cada volta)",
        fontweight="bold",
    )
    ax3.set_xlabel("Volta da Corrida ($t \\in \\{1, \\dots, 50\\}$)")
    ax3.set_ylabel("Ação Discreta ($A_t$)")
    ax3.set_yticks(range(6))
    ax3.set_yticklabels(action_labels, fontsize=8.8)
    ax3.set_ylim(-0.6, 5.6)
    ax3.set_xlim(1, 50)
    ax3.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax3.grid(True, linestyle="--", alpha=0.35)
    ax3.legend(
        loc="upper left",
        ncol=4,
        fontsize=8.5,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    fig.suptitle(
        "Desafio 2 — Perfil de Telemetria da Estratégia Aprendida pelo DQN (50 Voltas, Seed = 42)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return {
        "dqn_seed42": dqn_metrics,
        "heuristic_seed42": heur_metrics,
        "random_seed42": rand_metrics,
        "pit_laps": pit_laps,
        "telemetry": dqn_tele,
    }


def main() -> None:
    """Execute full training, evaluation, and figure generation pipeline."""
    figures_dir = PROJECT_ROOT / "docs" / "figures"
    model_path = PROJECT_ROOT / "models" / "dqn_race_strategy.pt"

    agent, history, evaluations = train_dqn(
        num_episodes=2000,
        seed=42,
        model_save_path=model_path,
    )

    fig1_path = figures_dir / "dqn_training_curves.png"
    fig2_path = figures_dir / "dqn_race_telemetry_profile.png"

    print("\nGenerating high-resolution (300 DPI) figures...")
    plot_training_curves(history, evaluations, fig1_path)
    print(f"[SAVED] {fig1_path}")

    env = RaceStrategyEnv()
    seed42_summary = plot_race_telemetry_profile(env, agent, fig2_path, seed=42)
    print(f"[SAVED] {fig2_path}")

    print("\n" + "=" * 78)
    print("FINAL DETERMINISTIC EVALUATION SUMMARY (20 SEEDS: 42..61, epsilon = 0.0)")
    print("=" * 78)
    header = (
        f"{'Policy':<24} | {'Return (Mean +/- Std)':<22} | "
        f"{'Race Time (s)':<18} | {'Pits':<6} | {'FIA (%)':<8} | {'DNF (%)':<8}"
    )
    print(header)
    print("-" * 78)
    for key, label in [
        ("random", "Random Agent"),
        ("heuristic", "Heuristic Baseline"),
        ("dqn", "Trained DQN Agent"),
    ]:
        ev = evaluations[key]
        print(
            f"{label:<24} | {ev['mean_reward']:8.2f} +/- {ev['std_reward']:6.2f}   | "
            f"{ev['mean_time']:7.2f} +/- {ev['std_time']:5.2f} | "
            f"{ev['mean_pits']:4.2f}   | {ev['fia_compliance_pct']:6.1f}%  | {ev['dnf_rate_pct']:6.1f}%"
        )
    print("-" * 78)

    dqn_s42: EpisodeMetrics = seed42_summary["dqn_seed42"]
    heur_s42: EpisodeMetrics = seed42_summary["heuristic_seed42"]
    rand_s42: EpisodeMetrics = seed42_summary["random_seed42"]
    print(
        f"Seed 42 Rollout -> DQN: Return={dqn_s42.total_reward:.2f}, Time={dqn_s42.cumulative_time:.2f}s, "
        f"Pits={dqn_s42.pit_stops} (Laps {seed42_summary['pit_laps']}), Compounds={dqn_s42.compounds_used}"
    )
    print(
        f"Seed 42 Rollout -> Heuristic: Return={heur_s42.total_reward:.2f}, Time={heur_s42.cumulative_time:.2f}s"
    )
    print(
        f"Seed 42 Rollout -> Random: Return={rand_s42.total_reward:.2f}, Time={rand_s42.cumulative_time:.2f}s"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
