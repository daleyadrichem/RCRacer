"""rc_racer.environment.collision

Fast, deterministic collision checks for the Environment layer.

The collision system is intentionally simple, deterministic, and cheap:

- Track boundary checks using :meth:`rc_racer.core.track.Track.is_inside`.
- Optional rectangular vehicle footprint checks, implemented by verifying that
  all footprint corners lie inside the track.

This module must remain independent of controllers and the GUI. The environment
may call these checks every step at a fixed timestep.

See Also
--------
rc_racer.environment.environment
    The Gym-like environment that uses this collision checker.

Notes
-----
This module follows the project's architectural constraints:

- No asynchronous execution
- No randomness
- No controller logic
- No GUI dependencies

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rc_racer.core.state import State, StateArray
from rc_racer.core.track import Track

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CollisionConfig:
    """Configuration for collision checking.

    Parameters
    ----------
    use_footprint : bool
        If True, check a rectangular vehicle footprint (corners) against the
        track boundaries. If False, only the reference point (state x,y) is
        tested.
    body_length : float
        Total vehicle body length in meters.
    body_width : float
        Total vehicle body width in meters.
    wheelbase : float
        Wheelbase in meters. Only used when ``origin_at_rear_axle=True`` to
        shift the body reference from rear axle to approximate vehicle center.
    rear_axle_ratio : float
        Fraction of wheelbase from rear axle to the vehicle center (or CG).
        Typical kinematic bicycle models store this as ``lr / (lf + lr)``.
    origin_at_rear_axle : bool
        If True, interpret ``State.x, State.y`` as the rear axle position.
        The footprint is then centered at an estimated vehicle center located
        ``rear_axle_ratio * wheelbase`` meters ahead along the heading.
        If False, interpret ``State.x, State.y`` as the vehicle center.
    margin : float
        Extra safety margin (meters) subtracted from half track width (i.e.,
        makes collision stricter). Useful to account for numerical errors or
        an approximate footprint.

    """

    use_footprint: bool = False

    body_length: float = 4.0
    body_width: float = 1.8

    wheelbase: float = 2.6
    rear_axle_ratio: float = 0.5

    origin_at_rear_axle: bool = True

    margin: float = 0.0


class CollisionChecker:
    """Fast collision checks against track boundaries.

    The primary collision condition for RCRacer is "off-track".

    Parameters
    ----------
    track : Track
        Immutable track geometry.
    config : CollisionConfig | None
        Collision configuration. If None, defaults are used.

    Notes
    -----
    - This class is deterministic.
    - This class performs no I/O and uses no randomness.
    """

    def __init__(self, track: Track, config: CollisionConfig | None = None) -> None:
        self._track: Track = track
        self._config: CollisionConfig = config if config is not None else CollisionConfig()

        # Cache half-width and margin-adjusted limit for fast point checks.
        half_width: float = 0.5 * float(track.width)
        self._inside_radius: Final[float] = max(0.0, half_width - float(self._config.margin))

    @property
    def track(self) -> Track:
        """Return the immutable track used by this checker."""
        return self._track

    @property
    def config(self) -> CollisionConfig:
        """Return the collision configuration."""
        return self._config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def is_collision(self, state: State) -> bool:
        """Check whether a single state is in collision (off-track).

        Parameters
        ----------
        state : State
            Scalar vehicle state.

        Returns
        -------
        bool
            True if the vehicle is considered in collision.
        """
        if not self._config.use_footprint:
            return not self._is_point_inside_track(float(state.x), float(state.y))

        corners: FloatArray = self._footprint_corners(state)
        # Collision if any corner is outside.
        for i in range(4):
            if not self._is_point_inside_track(float(corners[i, 0]), float(corners[i, 1])):
                return True
        return False

    def is_collision_array(self, states: StateArray) -> BoolArray:
        """Vectorized collision check for a batch of states.

        Parameters
        ----------
        states : StateArray
            Batched vehicle states.

        Returns
        -------
        ndarray of shape (N,), dtype=bool
            Boolean array where True indicates collision.

        Notes
        -----
        The track API is currently scalar (``project`` / ``is_inside``), so this
        method uses a tight Python loop over the batch. This keeps the contract
        deterministic and avoids any dependence on external geometry libraries.
        """
        n: int = int(states.batch_size)
        out: BoolArray = np.zeros((n,), dtype=np.bool_)

        if not self._config.use_footprint:
            for i in range(n):
                out[i] = not self._is_point_inside_track(float(states.x[i]), float(states.y[i]))
            return out

        # Footprint mode: compute per-state corners and test.
        for i in range(n):
            s = State(
                x=float(states.x[i]),
                y=float(states.y[i]),
                heading=float(states.heading[i]),
                velocity=float(states.velocity[i]),
                steering_angle=float(states.steering_angle[i]),
                progress_s=float(states.progress_s[i]),
            )
            out[i] = self.is_collision(s)
        return out

    # ---------------------------------------------------------------------
    # Internal geometry
    # ---------------------------------------------------------------------

    def _is_point_inside_track(self, x: float, y: float) -> bool:
        """Check point-in-track using a width-based distance test.

        Parameters
        ----------
        x : float
            World x-position.
        y : float
            World y-position.

        Returns
        -------
        bool
            True if the point is inside the track boundaries.

        Notes
        -----
        Using :meth:`Track.project` and a distance-to-centerline check is often
        faster and numerically robust.
        """
        pos: FloatArray = np.asarray([x, y], dtype=np.float64)
        _s, proj = self._track.project(pos)
        d: float = float(np.hypot(pos[0] - proj[0], pos[1] - proj[1]))
        return d <= self._inside_radius

    def _footprint_corners(self, state: State) -> FloatArray:
        """Compute rectangular footprint corners in world coordinates.

        Parameters
        ----------
        state : State
            Vehicle state.

        Returns
        -------
        ndarray of shape (4, 2)
            Corners ordered counter-clockwise.

        Notes
        -----
        The footprint is a simple rectangle aligned with the vehicle heading.
        It is intended as a fast approximation.
        """
        # Determine rectangle center.
        cx: float = float(state.x)
        cy: float = float(state.y)
        if self._config.origin_at_rear_axle:
            shift: float = float(self._config.rear_axle_ratio) * float(self._config.wheelbase)
            cx = cx + shift * float(np.cos(state.heading))
            cy = cy + shift * float(np.sin(state.heading))

        half_l: float = 0.5 * float(self._config.body_length)
        half_w: float = 0.5 * float(self._config.body_width)

        # Local-frame corners (vehicle frame): (forward, left)
        local: FloatArray = np.asarray(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ],
            dtype=np.float64,
        )

        c: float = float(np.cos(state.heading))
        s: float = float(np.sin(state.heading))
        rot: FloatArray = np.asarray([[c, -s], [s, c]], dtype=np.float64)

        world: FloatArray = (local @ rot.T) + np.asarray([cx, cy], dtype=np.float64)
        return world
