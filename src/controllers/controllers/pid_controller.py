"""
pid_controller.py

Multi-loop PID controller for track following.

Architecture Layer
------------------
AGENTS Layer

Responsibilities
----------------
- Lateral centerline tracking
- Heading alignment control
- Velocity regulation

Control Structure
-----------------
Steering rate command:
    u_steer =
        - (PID_lateral
           + PID_heading)

Acceleration command:
    u_acc =
        PID_speed

Each loop has independent (kp, ki, kd).

Notes
-----
- Fully type hinted.
- NumPy docstring style.
- Deterministic and stateless except for integrators.
- Does NOT control timestep (architecture rule).
- PIDConfig implements GenomeInterface.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Type

import json
import numpy as np
from numpy.typing import NDArray

from rc_racer.agents.base_controller import BaseController
from controllers.utils.genome_interface import GenomeInterface
from rc_racer.core.state import State
from rc_racer.core.track import Track


FloatArray = NDArray[np.float64]


# ================================================================
# Configuration (Genome Implementation)
# ================================================================


@dataclass(frozen=True)
class PIDConfig(GenomeInterface):
    """
    Configuration for multi-loop PID controller.

    This class also acts as the evolutionary genome.

    Parameters
    ----------
    kp_lat, ki_lat, kd_lat : float
        Lateral PID gains.
    kp_head, ki_head, kd_head : float
        Heading PID gains.
    kp_speed, ki_speed, kd_speed : float
        Speed PID gains.
    target_velocity : float
        Desired longitudinal velocity.
    """

    kp_lat: float
    ki_lat: float
    kd_lat: float

    kp_head: float
    ki_head: float
    kd_head: float

    kp_speed: float
    ki_speed: float
    kd_speed: float

    target_velocity: float

    # ============================================================
    # GenomeInterface Implementation
    # ============================================================

    @classmethod
    def genome_size(cls) -> int:
        """
        Return genome dimensionality.

        Returns
        -------
        int
        """
        return 10

    @classmethod
    def decode(cls: Type["PIDConfig"], z: FloatArray) -> "PIDConfig":
        """
        Decode raw genome vector using bounded sigmoid mapping.

        Parameters
        ----------
        z : ndarray

        Returns
        -------
        PIDConfig
        """

        # Bounds for each parameter
        LOW = np.array(
            [
                -12.0, -3.0, -6.0,   # lateral
                -12.0, -3.0, -6.0,   # heading
                -20.0, -5.0, -10.0,   # speed
                2.0,             # target_velocity
            ],
            dtype=np.float64,
        )

        HIGH = np.array(
            [
                12.0, 3.0, 6.0,
                12.0, 3.0, 6.0,
                20.0, 5.0, 10.0,
                30.0,
            ],
            dtype=np.float64,
        )

        z = np.clip(z, -60.0, 60.0)
        u = 1.0 / (1.0 + np.exp(-z))
        x = LOW + (HIGH - LOW) * u

        return cls(*x.tolist())

    @classmethod
    def from_dict(
        cls: Type["PIDConfig"],
        data: Dict[str, Any],
    ) -> "PIDConfig":
        """
        Construct config from dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        PIDConfig
        """
        return cls(**{k: float(v) for k, v in data.items()})

    def serialize(self) -> Dict[str, float]:
        """
        Serialize genome to dictionary.

        Returns
        -------
        dict[str, float]
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize genome to JSON string.

        Parameters
        ----------
        indent : int

        Returns
        -------
        str
        """
        return json.dumps(self.serialize(), indent=indent)

    def __str__(self) -> str:
        """
        Pretty print for console output.
        """
        return (
            f"PIDConfig("
            f"lat=({self.kp_lat:.3f},{self.ki_lat:.3f},{self.kd_lat:.3f}), "
            f"head=({self.kp_head:.3f},{self.ki_head:.3f},{self.kd_head:.3f}), "
            f"speed=({self.kp_speed:.3f},{self.ki_speed:.3f},{self.kd_speed:.3f}), "
            f"v={self.target_velocity:.3f})"
        )


# ================================================================
# Controller
# ================================================================


class PIDLineFollower(BaseController):
    """
    Multi-loop PID controller for track following.
    """

    def __init__(
        self,
        track: Track,
        config: PIDConfig,
    ) -> None:
        super().__init__()
        self._track: Track = track
        self.config: PIDConfig = config

        # Lateral loop
        self._lat_integral = 0.0
        self._lat_prev = 0.0

        # Heading loop
        self._head_integral = 0.0
        self._head_prev = 0.0

        # Speed loop
        self._speed_integral = 0.0
        self._speed_prev = 0.0

        self._debug_last_values: tuple[float, float, float, float, float, float] = (
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )

    # ============================================================
    # Lifecycle
    # ============================================================

    def reset(self) -> None:
        """Reset PID internal states."""
        self._lat_integral = 0.0
        self._lat_prev = 0.0

        self._head_integral = 0.0
        self._head_prev = 0.0

        self._speed_integral = 0.0
        self._speed_prev = 0.0

    # ============================================================
    # Core API
    # ============================================================

    def compute_action(
        self,
        state: State,
    ) -> Tuple[float, float]:

        position = np.array([state.x, state.y], dtype=np.float64)
        progress_s, projected_point = self._track.project(position)

        idx = np.searchsorted(self._track.arc_lengths, progress_s)
        idx = min(max(idx, 1), len(self._track.centerline) - 1)

        p0 = self._track.centerline[idx - 1]
        p1 = self._track.centerline[idx]
        tangent = p1 - p0
        tangent /= np.linalg.norm(tangent)

        normal = np.array([-tangent[1], tangent[0]])

        error_vec = position - projected_point
        lateral_error = float(np.dot(error_vec, normal))

        track_heading = float(np.arctan2(tangent[1], tangent[0]))
        heading_error = self._wrap_angle(track_heading - state.heading)

        speed_error = self.config.target_velocity - state.velocity

        # Lateral PID
        lat_derivative = lateral_error - self._lat_prev
        self._lat_integral += lateral_error
        self._lat_prev = lateral_error

        u_lat = (
            self.config.kp_lat * lateral_error
            + self.config.ki_lat * self._lat_integral
            + self.config.kd_lat * lat_derivative
        )

        # Heading PID
        head_derivative = heading_error - self._head_prev
        self._head_integral += heading_error
        self._head_prev = heading_error

        u_head = (
            self.config.kp_head * heading_error
            + self.config.ki_head * self._head_integral
            + self.config.kd_head * head_derivative
        )

        # Speed PID
        speed_derivative = speed_error - self._speed_prev
        self._speed_integral += speed_error
        self._speed_prev = speed_error

        acceleration = (
            self.config.kp_speed * speed_error
            + self.config.ki_speed * self._speed_integral
            + self.config.kd_speed * speed_derivative
        )

        steering_rate = -(u_lat + u_head)

        self._debug_last_values = (
            float(acceleration),
            float(u_lat),
            float(u_head),
            float(speed_error),
            float(heading_error),
            float(lateral_error),
        )

        return float(acceleration), float(steering_rate)

    # ============================================================
    # Debug
    # ============================================================

    @property
    def debug_values(self) -> tuple[float, float, float, float, float, float]:
        """
        Return last computed debug values.

        Returns
        -------
        tuple of float
            (
                acceleration,
                u_lat,
                u_head,
                speed_error,
                heading_error,
                lateral_error
            )
        """
        return self._debug_last_values


    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi
