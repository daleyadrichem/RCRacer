"""
reward.py

Deterministic reward system for RC Racer.
"""

from __future__ import annotations

from dataclasses import dataclass

from rc_racer.core.state import State


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class RewardConfig:
    """
    Reward configuration parameters.

    Parameters
    ----------
    progress_weight : float
        Reward multiplier for forward progress along track (delta s).

    off_track_penalty : float
        Penalty applied when vehicle is outside track boundaries.

    time_penalty : float
        Constant penalty applied each timestep.

    finish_bonus : float
        Bonus reward applied upon successful lap completion.
    """

    progress_weight: float
    off_track_penalty: float
    time_penalty: float
    finish_bonus: float


# ================================================================
# REWARD SYSTEM
# ================================================================


class RewardSystem:
    """
    Stateless deterministic reward calculator.

    Features
    --------
    - Progress-based reward
    - Off-track penalty
    - Per-step time penalty
    - Lap completion bonus

    Notes
    -----
    - Does NOT mutate state.
    - Does NOT track internal episode state.
    - Environment is responsible for passing
      previous and current states.
    """

    def __init__(self, config: RewardConfig) -> None:
        """
        Initialize reward system.

        Parameters
        ----------
        config : RewardConfig
            Immutable reward configuration.
        """
        self._config: RewardConfig = config

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def compute(
        self,
        previous_state: State,
        current_state: State,
        *,
        is_off_track: bool,
        lap_completed: bool,
    ) -> float:
        """
        Compute scalar reward for a single environment step.

        Parameters
        ----------
        previous_state : State
            State before transition.

        current_state : State
            State after transition.

        is_off_track : bool
            Whether vehicle is outside track boundaries.

        lap_completed : bool
            Whether lap was completed this step.

        Returns
        -------
        float
            Deterministic scalar reward.
        """
        reward: float = 0.0

        # --------------------------------------------------------
        # Progress Reward
        # --------------------------------------------------------
        delta_progress: float = (
            current_state.progress_s - previous_state.progress_s
        )

        reward += self._config.progress_weight * delta_progress

        # --------------------------------------------------------
        # Time Penalty
        # --------------------------------------------------------
        reward -= self._config.time_penalty

        # --------------------------------------------------------
        # Off-Track Penalty
        # --------------------------------------------------------
        if is_off_track:
            reward -= self._config.off_track_penalty

        # --------------------------------------------------------
        # Lap Completion Bonus
        # --------------------------------------------------------
        if lap_completed:
            reward += self._config.finish_bonus

        return reward
