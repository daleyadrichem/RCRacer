"""
termination.py

Episode termination conditions for the racing environment.

Termination Conditions
----------------------
• Lap completed
• Collision
• Timeout (max steps)
• Reverse driving
"""

from __future__ import annotations

from dataclasses import dataclass

from rc_racer.core.state import State


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class TerminationConfig:
    """
    Configuration for episode termination.

    Parameters
    ----------
    max_steps : int
        Maximum allowed environment steps.
    allow_reverse : bool
        If False, episode terminates when progress decreases.
    progress_epsilon : float
        Small tolerance when detecting reverse motion.
    """

    max_steps: int
    allow_reverse: bool = False
    progress_epsilon: float = 1e-6


# ================================================================
# TERMINATION CONDITION
# ================================================================


class TerminationCondition:
    """
    Deterministic episode termination logic.

    Notes
    -----
    - Stateless with respect to physics.
    - Tracks minimal episode-level counters only.
    - Must be reset at episode start.
    """

    def __init__(
        self,
        total_track_length: float,
        config: TerminationConfig,
    ) -> None:
        """
        Initialize termination condition.

        Parameters
        ----------
        total_track_length : float
            Total arc-length of the track.
        config : TerminationConfig
            Termination configuration.
        """
        self._total_track_length: float = float(total_track_length)
        self._config: TerminationConfig = config

        self._step_count: int = 0
        self._previous_progress: float = 0.0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def reset(self, initial_state: State) -> None:
        """
        Reset termination tracking for new episode.

        Parameters
        ----------
        initial_state : State
            Initial vehicle state.
        """
        self._step_count = 0
        self._previous_progress = initial_state.progress_s

    def check(
        self,
        state: State,
        *,
        collision: bool,
    ) -> bool:
        """
        Evaluate termination condition.

        Parameters
        ----------
        state : State
            Current vehicle state.
        collision : bool
            Whether a collision occurred.

        Returns
        -------
        bool
            True if episode should terminate.
        """
        self._step_count += 1

        # --------------------------------------------------------
        # 1. Collision
        # --------------------------------------------------------
        if collision:
            return True

        # --------------------------------------------------------
        # 2. Lap Completed
        # --------------------------------------------------------
        if state.progress_s >= self._total_track_length:
            return True

        # --------------------------------------------------------
        # 3. Timeout
        # --------------------------------------------------------
        if self._step_count >= self._config.max_steps:
            return True

        # --------------------------------------------------------
        # 4. Reverse Driving
        # --------------------------------------------------------
        if not self._config.allow_reverse:
            if (
                state.progress_s
                < self._previous_progress - self._config.progress_epsilon
            ):
                return True

        self._previous_progress = state.progress_s
        return False

    # ------------------------------------------------------------
    # Introspection (Optional)
    # ------------------------------------------------------------

    @property
    def step_count(self) -> int:
        """
        Current episode step count.

        Returns
        -------
        int
        """
        return self._step_count
