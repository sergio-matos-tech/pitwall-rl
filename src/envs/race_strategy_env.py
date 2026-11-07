"""Race Strategy Environment (RaceStrategy-v0).

A custom Gymnasium environment modeling a Formula 1 / endurance race strategy
problem as a Markov Decision Process (MDP).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt


class TireCompound(IntEnum):
    """Enumeration of available tire compounds."""

    SOFT = 0
    MEDIUM = 1
    HARD = 2


class PaceMode(IntEnum):
    """Enumeration of driving pace modes."""

    CONSERVE = 0
    BALANCED = 1
    PUSH = 2


class RaceAction(IntEnum):
    """Enumeration of agent action choices per lap."""

    PUSH = 0
    BALANCED = 1
    CONSERVE = 2
    PIT_SOFT = 3
    PIT_MEDIUM = 4
    PIT_HARD = 5


@dataclass(frozen=True)
class TireProperties:
    """Physical specifications and degradation characteristics of a tire compound.

    Attributes:
        name: Human-readable compound designation.
        base_lap_delta: Base lap time offset relative to reference (seconds). Negative is faster.
        base_wear_rate: Wear increment per nominal lap at balanced pace.
        cliff_threshold: Wear level (0.0 to 1.0) beyond which degradation accelerates exponentially.
        cliff_exponent: Non-linear steepness exponent for thermal/surface degradation cliff.
        cliff_max_penalty: Maximum lap time penalty incurred deep into the cliff (seconds).
        linear_wear_penalty: Gradual wear penalty before hitting the cliff (seconds).
    """

    name: str
    base_lap_delta: float
    base_wear_rate: float
    cliff_threshold: float
    cliff_exponent: float = 3.5
    cliff_max_penalty: float = 8.0
    linear_wear_penalty: float = 1.5


@dataclass(frozen=True)
class RaceConfig:
    """Frozen configuration holding all physical and operational parameters.

    Adheres to the Single Responsibility Principle (SRP) by separating physics
    and hyperparameters from the MDP environment mechanics.
    """

    total_laps: int = 50
    base_lap_time: float = 90.0  # seconds
    pit_lane_loss: float = 22.0  # seconds lost during pit stop
    initial_fuel: float = 100.0  # kg
    fuel_burn_base: float = 1.8  # kg per lap at balanced pace
    fuel_time_penalty_per_kg: float = 0.033  # seconds slower per kg of onboard fuel

    # Pace modifiers: (lap_delta_seconds, wear_factor, fuel_factor)
    pace_modifiers: dict[PaceMode, tuple[float, float, float]] = field(
        default_factory=lambda: {
            PaceMode.CONSERVE: (0.90, 0.70, 0.90),
            PaceMode.BALANCED: (0.00, 1.00, 1.00),
            PaceMode.PUSH: (-0.90, 1.55, 1.18),
        }
    )

    # Compound specifications
    compounds: dict[TireCompound, TireProperties] = field(
        default_factory=lambda: {
            TireCompound.SOFT: TireProperties(
                name="Soft",
                base_lap_delta=-1.20,
                base_wear_rate=0.035,
                cliff_threshold=0.65,
                cliff_exponent=3.5,
                cliff_max_penalty=8.5,
                linear_wear_penalty=1.8,
            ),
            TireCompound.MEDIUM: TireProperties(
                name="Medium",
                base_lap_delta=0.00,
                base_wear_rate=0.022,
                cliff_threshold=0.72,
                cliff_exponent=3.5,
                cliff_max_penalty=7.5,
                linear_wear_penalty=1.4,
            ),
            TireCompound.HARD: TireProperties(
                name="Hard",
                base_lap_delta=1.00,
                base_wear_rate=0.014,
                cliff_threshold=0.80,
                cliff_exponent=3.5,
                cliff_max_penalty=6.5,
                linear_wear_penalty=1.1,
            ),
        }
    )

    # Noise & safety margins
    lap_noise_std: float = 0.15
    wear_noise_std: float = 0.001
    fuel_noise_std: float = 0.04
    puncture_wear_threshold: float = 1.0
    puncture_time_penalty: float = 60.0
    puncture_reward_penalty: float = 50.0

    # Rules
    mandatory_compound_change: bool = True
    illegal_strategy_penalty: float = 100.0
    finish_bonus: float = 20.0


class RaceStrategyEnv(gym.Env[npt.NDArray[np.float32], int]):
    """Gymnasium v1.0+ Custom Environment for Race Strategy Optimization.

    Markov Decision Process (MDP) specification:
    -------------------------------------------------
    - State Space S: 7-dimensional Box space with float32 precision.
        [0] lap_progress (0.0 to 1.0)
        [1] tire_wear (0.0 to 1.0, where 0.0 is brand new, 1.0 is puncture)
        [2] tire_compound (0.0 = Soft, 1.0 = Medium, 2.0 = Hard)
        [3] fuel_ratio (fuel_kg / initial_fuel in [0.0, 1.0])
        [4] pit_stops_count (normalized or absolute count)
        [5] last_lap_delta (last_lap_time - base_lap_time in seconds)
        [6] unique_compounds_used (count of unique compounds utilized so far)

    - Action Space A: Discrete(6)
        0: PUSH (Stay out, aggressive pace)
        1: BALANCED (Stay out, nominal pace)
        2: CONSERVE (Stay out, tire/fuel saving)
        3: PIT_SOFT (Box this lap, fit fresh Soft tires)
        4: PIT_MEDIUM (Box this lap, fit fresh Medium tires)
        5: PIT_HARD (Box this lap, fit fresh Hard tires)

    - Reward Function R:
        R_t = -(T_lap - T_base) - Penalty_puncture + Bonus_terminal
        Minimizing cumulative race time corresponds directly to maximizing
        cumulative returns.
    """

    metadata: dict[str, Any] = {
        "render_modes": ["human", "ansi"],
        "render_fps": 10,
    }

    def __init__(
        self,
        config: RaceConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Initialize the RaceStrategy environment.

        Args:
            config: Optional custom RaceConfig instance. If None, default parameters are used.
            render_mode: Desired render mode ('human', 'ansi', or None).
        """
        super().__init__()
        self.config = config if config is not None else RaceConfig()
        self.render_mode = render_mode

        # Action space: 6 discrete choices
        self.action_space = spaces.Discrete(len(RaceAction))

        # Observation space: 7-element vector [dtype=np.float32]
        # [lap_progress, tire_wear, compound, fuel_ratio, pit_stops, last_lap_delta, compounds_count]
        lows = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -10.0, 1.0], dtype=np.float32)
        highs = np.array([1.0, 1.0, 2.0, 1.0, 10.0, 150.0, 3.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=lows,
            high=highs,
            shape=(7,),
            dtype=np.float32,
        )

        # Internal simulation states
        self._current_lap: int = 0
        self._tire_wear: float = 0.0
        self._current_compound: TireCompound = TireCompound.MEDIUM
        self._fuel_kg: float = self.config.initial_fuel
        self._pit_stops_count: int = 0
        self._last_lap_time: float = self.config.base_lap_time
        self._compounds_used: set[TireCompound] = set()
        self._cumulative_time: float = 0.0
        self._lap_history: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the environment to its initial race state.

        Args:
            seed: Optional integer seed for reproducible PRNG.
            options: Optional dictionary containing custom initialization parameters:
                     - 'starting_compound': TireCompound to begin the race on.

        Returns:
            A tuple of (initial_observation, info_dictionary).
        """
        super().reset(seed=seed)

        starting_compound = TireCompound.MEDIUM
        if options and "starting_compound" in options:
            starting_compound = TireCompound(options["starting_compound"])

        self._current_lap = 0
        self._tire_wear = 0.0
        self._current_compound = starting_compound
        self._fuel_kg = float(self.config.initial_fuel)
        self._pit_stops_count = 0
        self._last_lap_time = float(self.config.base_lap_time)
        self._compounds_used = {starting_compound}
        self._cumulative_time = 0.0
        self._lap_history.clear()

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(
        self,
        action: int | np.integer | RaceAction,
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance the simulation by executing one lap with the chosen action.

        Args:
            action: The selected action (RaceAction enum or integer 0..5).

        Returns:
            A 5-tuple of (observation, reward, terminated, truncated, info).
        """
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action!r} passed to step. "
                f"Action space is {self.action_space}."
            )

        act = RaceAction(int(action))
        is_pit_stop = act in (
            RaceAction.PIT_SOFT,
            RaceAction.PIT_MEDIUM,
            RaceAction.PIT_HARD,
        )

        # 1. Determine pace mode and compound switch
        if is_pit_stop:
            pace_mode = PaceMode.BALANCED  # in-lap driven at standard pace
            if act == RaceAction.PIT_SOFT:
                new_compound = TireCompound.SOFT
            elif act == RaceAction.PIT_MEDIUM:
                new_compound = TireCompound.MEDIUM
            else:
                new_compound = TireCompound.HARD
        else:
            if act == RaceAction.PUSH:
                pace_mode = PaceMode.PUSH
            elif act == RaceAction.CONSERVE:
                pace_mode = PaceMode.CONSERVE
            else:
                pace_mode = PaceMode.BALANCED
            new_compound = self._current_compound

        pace_delta, pace_wear_mult, pace_fuel_mult = self.config.pace_modifiers[
            pace_mode
        ]
        active_compound_props = self.config.compounds[self._current_compound]

        # 2. Calculate tire degradation penalty for this lap
        # Non-linear tire cliff modeling
        wear = self._tire_wear
        cliff = active_compound_props.cliff_threshold
        linear_penalty = active_compound_props.linear_wear_penalty * wear

        cliff_penalty = 0.0
        if wear > cliff:
            relative_cliff = (wear - cliff) / max(1e-5, (1.0 - cliff))
            cliff_penalty = (
                active_compound_props.cliff_max_penalty
                * (relative_cliff**active_compound_props.cliff_exponent)
            )

        tire_penalty = linear_penalty + cliff_penalty

        # Check for catastrophic tire failure (puncture)
        is_puncture = wear >= self.config.puncture_wear_threshold
        puncture_lap_penalty = (
            self.config.puncture_time_penalty if is_puncture else 0.0
        )

        # 3. Calculate fuel penalty and fuel consumption
        fuel_time_penalty = self._fuel_kg * self.config.fuel_time_penalty_per_kg

        fuel_noise = float(self.np_random.normal(0.0, self.config.fuel_noise_std))
        fuel_consumed = max(
            0.5,
            self.config.fuel_burn_base * pace_fuel_mult + fuel_noise,
        )
        self._fuel_kg = max(0.0, self._fuel_kg - fuel_consumed)

        # 4. Pit stop time loss
        pit_time_loss = self.config.pit_lane_loss if is_pit_stop else 0.0

        # 5. Stochastic lap noise (track evolution, traffic, driver variance)
        lap_noise = float(self.np_random.normal(0.0, self.config.lap_noise_std))

        # 6. Total lap time synthesis
        lap_time = (
            self.config.base_lap_time
            + active_compound_props.base_lap_delta
            + pace_delta
            + fuel_time_penalty
            + tire_penalty
            + puncture_lap_penalty
            + pit_time_loss
            + lap_noise
        )

        # Clamp minimum realistic lap time
        lap_time = max(70.0, lap_time)

        # 7. Update state variables
        self._current_lap += 1
        self._last_lap_time = lap_time
        self._cumulative_time += lap_time

        # Update tire wear for the next lap
        if is_pit_stop:
            self._tire_wear = 0.0
            self._current_compound = new_compound
            self._compounds_used.add(new_compound)
            self._pit_stops_count += 1
        else:
            # Accelerated wear as tire surface breaks down: (1 + 0.5 * w^2)
            wear_noise = float(
                self.np_random.normal(0.0, self.config.wear_noise_std)
            )
            wear_step = (
                active_compound_props.base_wear_rate
                * pace_wear_mult
                * (1.0 + 0.5 * (self._tire_wear**2))
                + wear_noise
            )
            self._tire_wear = float(
                np.clip(self._tire_wear + max(0.0, wear_step), 0.0, 1.0)
            )

        # 8. Termination and truncation conditions
        terminated = False
        truncated = False

        if is_puncture:
            # Race is terminated early due to unmanageable tire failure
            terminated = True
        elif self._current_lap >= self.config.total_laps:
            terminated = True

        # 9. Reward formulation
        # Dense lap-by-lap objective: reward = -(T_lap - T_base)
        lap_reward = -(lap_time - self.config.base_lap_time)

        # Puncture penalty
        puncture_penalty = (
            self.config.puncture_reward_penalty if is_puncture else 0.0
        )

        # Terminal bonuses / penalties
        terminal_reward = 0.0
        if terminated and not is_puncture:
            # Check mandatory compound usage rule
            if self.config.mandatory_compound_change and len(self._compounds_used) < 2:
                terminal_reward -= self.config.illegal_strategy_penalty
            else:
                terminal_reward += self.config.finish_bonus

        reward = float(lap_reward - puncture_penalty + terminal_reward)

        observation = self._get_observation()
        info = self._get_info()
        info.update(
            {
                "lap_time": lap_time,
                "lap_reward": lap_reward,
                "is_pit_stop": is_pit_stop,
                "is_puncture": is_puncture,
                "pace_mode": pace_mode.name,
                "tire_penalty": tire_penalty,
                "cliff_penalty": cliff_penalty,
                "fuel_penalty": fuel_time_penalty,
            }
        )

        # Record telemetry history
        self._lap_history.append(
            {
                "lap": self._current_lap,
                "action": act.name,
                "compound": self._current_compound.name,
                "wear": self._tire_wear,
                "fuel_kg": self._fuel_kg,
                "lap_time": lap_time,
                "cumulative_time": self._cumulative_time,
                "reward": reward,
            }
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> npt.NDArray[np.float32]:
        """Vectorize internal states into a normalized float32 observation array."""
        lap_progress = float(self._current_lap) / float(self.config.total_laps)
        tire_wear = float(self._tire_wear)
        compound = float(self._current_compound.value)
        fuel_ratio = float(self._fuel_kg / self.config.initial_fuel)
        pit_stops = float(self._pit_stops_count)
        last_lap_delta = float(self._last_lap_time - self.config.base_lap_time)
        unique_compounds = float(len(self._compounds_used))

        return np.array(
            [
                lap_progress,
                tire_wear,
                compound,
                fuel_ratio,
                pit_stops,
                last_lap_delta,
                unique_compounds,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict[str, Any]:
        """Generate diagnostic metadata and telemetry."""
        return {
            "current_lap": self._current_lap,
            "total_laps": self.config.total_laps,
            "tire_wear": self._tire_wear,
            "current_compound": self._current_compound.name,
            "fuel_kg": self._fuel_kg,
            "pit_stops_count": self._pit_stops_count,
            "cumulative_time": self._cumulative_time,
            "compounds_used": [c.name for c in self._compounds_used],
            "valid_strategy": len(self._compounds_used) >= 2
            if self.config.mandatory_compound_change
            else True,
        }

    def render(self) -> str | None:
        """Render the race telemetry dashboard.

        Returns string if render_mode is 'ansi', otherwise prints to stdout.
        """
        wear_pct = int(self._tire_wear * 100)
        fuel_pct = int((self._fuel_kg / self.config.initial_fuel) * 100)

        # ASCII wear bar
        filled_bars = int(self._tire_wear * 10)
        wear_bar = "#" * filled_bars + "-" * (10 - filled_bars)

        status_text = (
            f"[Lap {self._current_lap:02d}/{self.config.total_laps:02d}] "
            f"Tire: {self._current_compound.name:<6} Wear: [{wear_bar}] {wear_pct:3d}% | "
            f"Fuel: {self._fuel_kg:5.1f}kg ({fuel_pct:3d}%) | "
            f"Pits: {self._pit_stops_count} | "
            f"Last Lap: {self._last_lap_time:6.2f}s | "
            f"Total: {self._cumulative_time:7.2f}s"
        )

        if self.render_mode == "human":
            print(status_text)
            return None
        return status_text

    def close(self) -> None:
        """Clean up environment resources."""
        self._lap_history.clear()
        super().close()
