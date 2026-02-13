"""
example_pid.py

Example PID controller that follows the track centerline.

This controller:
- Projects vehicle position onto the track centerline
- Computes lateral cross-track error
- Uses PID steering control to minimize lateral error
- Uses simple proportional speed control to maintain target velocity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rc_racer.agents.base_controller import BaseController
from rc_racer.core.state import State
from rc_racer.core.track import Track


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class PIDConfig:
    """
    Configuration for PID line-following controller.

    Parameters
    ----------
    kp : float
        Proportional gain for lateral error.
    ki : float
        Integral gain for lateral error.
    kd : float
        Derivative gain for lateral error.
    target_velocity : float
        Desired forward velocity [m/s].
    speed_kp : float
        Proportional gain for velocity tracking.
    """

    kp: float = 3.0
    ki: float = 0.0
    kd: float = 0.5
    target_velocity: float = 8.0
    speed_kp: float = 1.5


# ================================================================
# CONTROLLER
# ================================================================


class PIDLineFollower(BaseController):
    """
    Simple PID controller for centerline following.

    Strategy
    --------
    1. Project vehicle position onto track centerline.
    2. Compute signed lateral error.
    3. Apply PID to steering rate.
    4. Apply proportional control to velocity.

    Notes
    -----
    - Deterministic.
    - Does not modify state.
    - Does not control timestep.
    """

    def __init__(
        self,
        track: Track,
        config: PIDConfig | None = None,
    ) -> None:
        """
        Initialize PID controller.

        Parameters
        ----------
        track : Track
            Immutable track geometry.
        config : PIDConfig | None
            PID tuning parameters.
        """
        self._track: Track = track
        self._config: PIDConfig = config or PIDConfig()

        self._integral_error: float = 0.0
        self._previous_error: float = 0.0

    # ------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset controller internal memory.
        """
        self._integral_error = 0.0
        self._previous_error = 0.0

    # ------------------------------------------------------------

    def compute_action(
        self,
        state: State,
    ) -> Tuple[float, float]:
        """
        Compute acceleration and steering rate.

        Parameters
        ----------
        state : State
            Immutable vehicle state.

        Returns
        -------
        tuple[float, float]
            (acceleration_command, steering_rate_command)
        """
        # --------------------------------------------------------
        # Project onto track
        # --------------------------------------------------------
        position = np.array([state.x, state.y], dtype=np.float64)
        progress_s, proj = self._track.project(position)

        # --------------------------------------------------------
        # Compute signed lateral error
        # --------------------------------------------------------
        center_vec = proj - position

        # Track tangent direction
        idx = np.searchsorted(self._track.arc_lengths, progress_s)
        idx = min(max(idx, 1), len(self._track.centerline) - 1)

        p0 = self._track.centerline[idx - 1]
        p1 = self._track.centerline[idx]
        tangent = p1 - p0
        tangent /= np.linalg.norm(tangent)

        # Normal vector
        normal = np.array([-tangent[1], tangent[0]])

        lateral_error = float(np.dot(center_vec, normal))

        # --------------------------------------------------------
        # PID steering control
        # --------------------------------------------------------
        self._integral_error += lateral_error
        derivative = lateral_error - self._previous_error

        steering_rate = (
            self._config.kp * lateral_error
            + self._config.ki * self._integral_error
            + self._config.kd * derivative
        )

        self._previous_error = lateral_error

        # --------------------------------------------------------
        # Speed control
        # --------------------------------------------------------
        velocity_error = self._config.target_velocity - state.velocity
        acceleration = self._config.speed_kp * velocity_error

        return float(acceleration), float(steering_rate)
