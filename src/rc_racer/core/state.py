"""
state.py

Vehicle state definition for the racing simulation core.

This module defines:

- Immutable scalar State (single vehicle)
- Vectorized StateArray (batch vehicles)
- Serialization helpers for deterministic replay
- Validation constraints for physical correctness
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


# ============================================================
# Scalar State
# ============================================================


@dataclass(frozen=True)
class State:
    """
    Immutable vehicle state.

    Parameters
    ----------
    x : float
        Global x-position [m].
    y : float
        Global y-position [m].
    heading : float
        Heading angle [rad].
    velocity : float
        Longitudinal velocity [m/s]. Must be >= 0.
    steering_angle : float
        Steering angle [rad].
    progress_s : float
        Arc-length progress along track [m].

    Raises
    ------
    ValueError
        If velocity < 0.
    """

    x: float
    y: float
    heading: float
    velocity: float
    steering_angle: float
    progress_s: float

    def __post_init__(self) -> None:
        if self.velocity < 0.0:
            raise ValueError("Velocity must be non-negative.")

    # --------------------------------------------------------

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        """
        Convert state to tuple.

        Returns
        -------
        tuple of float
        """
        return (
            self.x,
            self.y,
            self.heading,
            self.velocity,
            self.steering_angle,
            self.progress_s,
        )

    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, float]:
        """
        Serialize state to dictionary.

        Returns
        -------
        dict
            JSON-safe representation.
        """
        return {
            "x": self.x,
            "y": self.y,
            "heading": self.heading,
            "velocity": self.velocity,
            "steering_angle": self.steering_angle,
            "progress_s": self.progress_s,
        }

    # --------------------------------------------------------

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> State:
        """
        Deserialize state from dictionary.

        Parameters
        ----------
        data : dict

        Returns
        -------
        State
        """
        return State(
            x=float(data["x"]),
            y=float(data["y"]),
            heading=float(data["heading"]),
            velocity=float(data["velocity"]),
            steering_angle=float(data["steering_angle"]),
            progress_s=float(data["progress_s"]),
        )

    # --------------------------------------------------------

    def copy_with(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        heading: float | None = None,
        velocity: float | None = None,
        steering_angle: float | None = None,
        progress_s: float | None = None,
    ) -> State:
        """
        Return new State with selected fields replaced.
        """
        return State(
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            heading=self.heading if heading is None else heading,
            velocity=self.velocity if velocity is None else velocity,
            steering_angle=self.steering_angle
            if steering_angle is None
            else steering_angle,
            progress_s=self.progress_s if progress_s is None else progress_s,
        )


# ============================================================
# Vectorized StateArray
# ============================================================


@dataclass(frozen=True)
class StateArray:
    """
    Vectorized vehicle state container.

    Designed for future vector_env.py for parallel stepping.

    All arrays must:
    - Have dtype float64
    - Have identical shape (N,)

    Parameters
    ----------
    x, y, heading, velocity, steering_angle, progress_s : ndarray

    Raises
    ------
    ValueError
        If shapes mismatch or velocity contains negative values.
    """

    x: FloatArray
    y: FloatArray
    heading: FloatArray
    velocity: FloatArray
    steering_angle: FloatArray
    progress_s: FloatArray

    def __post_init__(self) -> None:
        shapes = {
            self.x.shape,
            self.y.shape,
            self.heading.shape,
            self.velocity.shape,
            self.steering_angle.shape,
            self.progress_s.shape,
        }

        if len(shapes) != 1:
            raise ValueError("All arrays must have identical shape.")

        if np.any(self.velocity < 0.0):
            raise ValueError("Velocity values must be non-negative.")

        for arr in (
            self.x,
            self.y,
            self.heading,
            self.velocity,
            self.steering_angle,
            self.progress_s,
        ):
            if arr.dtype != np.float64:
                raise ValueError("All arrays must be float64.")

    # --------------------------------------------------------

    @property
    def batch_size(self) -> int:
        """
        Number of vehicles.

        Returns
        -------
        int
        """
        return self.x.shape[0]

    # --------------------------------------------------------

    def to_dict(self) -> Dict[str, list[float]]:
        """
        Serialize to JSON-safe dict.
        """
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "heading": self.heading.tolist(),
            "velocity": self.velocity.tolist(),
            "steering_angle": self.steering_angle.tolist(),
            "progress_s": self.progress_s.tolist(),
        }

    # --------------------------------------------------------

    @staticmethod
    def from_dict(data: Dict[str, Iterable[float]]) -> StateArray:
        """
        Deserialize from dictionary.
        """
        return StateArray(
            x=np.asarray(data["x"], dtype=np.float64),
            y=np.asarray(data["y"], dtype=np.float64),
            heading=np.asarray(data["heading"], dtype=np.float64),
            velocity=np.asarray(data["velocity"], dtype=np.float64),
            steering_angle=np.asarray(data["steering_angle"], dtype=np.float64),
            progress_s=np.asarray(data["progress_s"], dtype=np.float64),
        )
