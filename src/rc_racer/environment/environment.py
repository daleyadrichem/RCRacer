"""
environment.py

Deterministic racing environment implementation.

ENVIRONMENT Layer
-----------------
Implements a Gym-like synchronous API:

    reset(seed=None)
    step(action)

Contains:
    - Track
    - VehicleModel
    - CollisionChecker
    - RewardSystem
    - TerminationCondition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from rc_racer.core.state import State
from rc_racer.core.track import Track
from rc_racer.core.vehicle_model import VehicleModel
from rc_racer.environment.collision import CollisionChecker
from rc_racer.environment.reward import RewardSystem
from rc_racer.environment.termination import TerminationCondition

FloatArray = NDArray[np.float64]


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    Environment configuration container.

    Parameters
    ----------
    dt : float
        Fixed simulation timestep in seconds.
    """

    dt: float


# ================================================================
# ENVIRONMENT
# ================================================================


class Environment:
    """
    Deterministic racing environment.

    Notes
    -----
    - Synchronous stepping only.
    - Simulation clock is authoritative.
    - Controller never controls timestep.
    - All randomness must be seeded explicitly.
    - No shared mutable state.
    """

    # ------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------

    def __init__(
        self,
        track: Track,
        vehicle_model: VehicleModel,
        collision_checker: CollisionChecker,
        reward_system: RewardSystem,
        termination_condition: TerminationCondition,
        config: EnvironmentConfig,
    ) -> None:
        """
        Initialize environment.

        Parameters
        ----------
        track : Track
        vehicle_model : VehicleModel
        collision_checker : CollisionChecker
        reward_system : RewardSystem
        termination_condition : TerminationCondition
        config : EnvironmentConfig
        """
        self._track: Track = track
        self._vehicle_model: VehicleModel = vehicle_model
        self._collision_checker: CollisionChecker = collision_checker
        self._reward_system: RewardSystem = reward_system
        self._termination_condition: TerminationCondition = termination_condition
        self._config: EnvironmentConfig = config

        self._rng: np.random.Generator = np.random.default_rng()
        self._state: Optional[State] = None

    # ------------------------------------------------------------
    # PROPERTIES
    # ------------------------------------------------------------

    @property
    def state(self) -> State:
        """
        Current environment state.

        Returns
        -------
        State
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before accessing state.")
        return self._state

    @property
    def dt(self) -> float:
        """
        Fixed timestep.

        Returns
        -------
        float
        """
        return self._config.dt

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> State:
        """
        Reset environment to initial state.

        Parameters
        ----------
        seed : int | None
            Optional random seed for reproducibility.

        Returns
        -------
        State
            Initial vehicle state.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Deterministic initial placement:
        # Start at first centerline point
        start_xy: FloatArray = self._track.centerline[0]

        # Initial heading aligned with first segment
        if self._track.centerline.shape[0] > 1:
            delta = self._track.centerline[1] - self._track.centerline[0]
            heading = float(np.arctan2(delta[1], delta[0]))
        else:
            heading = 0.0

        initial_state = State(
            x=float(start_xy[0]),
            y=float(start_xy[1]),
            heading=heading,
            velocity=0.0,
            steering_angle=0.0,
            progress_s=0.0,
        )

        self._state = initial_state

        self._termination_condition.reset(initial_state)

        return initial_state

    # ------------------------------------------------------------
    # STEP
    # ------------------------------------------------------------

    def step(
        self,
        action: Tuple[float, float],
    ) -> Tuple[State, float, bool, dict]:
        """
        Advance simulation by one timestep.

        Parameters
        ----------
        action : tuple[float, float]
            (acceleration_command, steering_rate_command)

        Returns
        -------
        state : State
        reward : float
        done : bool
        info : dict
        """
        if self._state is None:
            raise RuntimeError("Environment must be reset before stepping.")

        previous_state: State = self._state

        # --------------------------------------------------------
        # Physics Step (Deterministic)
        # --------------------------------------------------------
        next_state = self._vehicle_model.step(
            previous_state,
            action,
            self._config.dt,
        )

        # --------------------------------------------------------
        # Progress Projection
        # --------------------------------------------------------
        progress_s, _ = self._track.project(
            np.array([next_state.x, next_state.y], dtype=np.float64)
        )

        next_state = next_state.copy_with(progress_s=progress_s)

        # --------------------------------------------------------
        # Collision Check
        # --------------------------------------------------------
        collision: bool = self._collision_checker.is_collision(next_state)

        # --------------------------------------------------------
        # Lap Completion Check
        # --------------------------------------------------------
        lap_completed: bool = progress_s >= self._track.total_length

        # --------------------------------------------------------
        # Reward
        # --------------------------------------------------------
        reward: float = self._reward_system.compute(
            previous_state,
            next_state,
            is_off_track=collision,
            lap_completed=lap_completed,
        )

        # --------------------------------------------------------
        # Termination
        # --------------------------------------------------------
        done: bool = self._termination_condition.check(
            next_state,
            collision=collision,
        )

        # --------------------------------------------------------
        # Commit State
        # --------------------------------------------------------
        self._state = next_state

        info: dict = {
            "collision": collision,
            "lap_completed": lap_completed,
            "step_count": self._termination_condition.step_count,
        }

        return next_state, reward, done, info
