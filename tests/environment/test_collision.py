"""
test_collision.py

Unit tests for rc_racer.environment.collision.

Covers
------
- Point-based collision detection
- Margin behavior
- Footprint collision detection
- Rear-axle origin shifting
- Vectorized collision checking

Architecture
------------
These tests exercise only the Environment-layer collision module.
They do NOT involve:
- Controllers
- Environment stepping
- GUI
- Randomness

All tests are deterministic.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rc_racer.core.state import State, StateArray
from rc_racer.core.track import Track
from rc_racer.environment.collision import CollisionChecker, CollisionConfig

FloatArray = NDArray[np.float64]


# ================================================================
# Helpers
# ================================================================


def _straight_track(width: float = 4.0) -> Track:
    """
    Create a simple straight track along the x-axis.

    Parameters
    ----------
    width : float
        Track width in meters.

    Returns
    -------
    Track
        Deterministic straight track.
    """
    centerline: FloatArray = np.array(
        [
            [0.0, 0.0],
            [100.0, 0.0],
        ],
        dtype=np.float64,
    )
    return Track(centerline=centerline, width=width)


def _state(
    x: float,
    y: float,
    heading: float = 0.0,
) -> State:
    """
    Create a minimal valid scalar State.

    Parameters
    ----------
    x : float
    y : float
    heading : float

    Returns
    -------
    State
    """
    return State(
        x=x,
        y=y,
        heading=heading,
        velocity=0.0,
        steering_angle=0.0,
        progress_s=0.0,
    )


# ================================================================
# Point-based collision tests
# ================================================================


def test_point_inside_track() -> None:
    """Point strictly inside track width should not collide."""
    track = _straight_track(width=4.0)
    checker = CollisionChecker(track)

    s = _state(10.0, 0.5)
    assert not checker.is_collision(s)


def test_point_outside_track() -> None:
    """Point outside half-width should collide."""
    track = _straight_track(width=4.0)
    checker = CollisionChecker(track)

    # half-width = 2.0
    s = _state(10.0, 2.5)
    assert checker.is_collision(s)


def test_margin_makes_collision_stricter() -> None:
    """Margin reduces effective inside radius."""
    track = _straight_track(width=4.0)

    # half-width = 2.0
    # margin=0.5 → effective radius = 1.5
    config = CollisionConfig(margin=0.5)
    checker = CollisionChecker(track, config=config)

    s = _state(10.0, 1.6)
    assert checker.is_collision(s)


# ================================================================
# Footprint collision tests
# ================================================================


def test_footprint_inside_track() -> None:
    """Footprint fully inside track should not collide."""
    track = _straight_track(width=6.0)

    config = CollisionConfig(
        use_footprint=True,
        body_length=2.0,
        body_width=1.0,
        origin_at_rear_axle=False,
    )

    checker = CollisionChecker(track, config=config)

    s = _state(10.0, 0.0)
    assert not checker.is_collision(s)


def test_footprint_outside_due_to_corner() -> None:
    """Footprint corner crossing boundary should collide."""
    track = _straight_track(width=4.0)

    config = CollisionConfig(
        use_footprint=True,
        body_length=4.0,
        body_width=2.0,
        origin_at_rear_axle=False,
    )

    checker = CollisionChecker(track, config=config)

    # Center near boundary; corner extends outside
    s = _state(10.0, 1.5)
    assert checker.is_collision(s)


def test_rear_axle_origin_shift() -> None:
    """
    Rear axle origin mode shifts footprint forward.
    Should remain inside for centered placement.
    """
    track = _straight_track(width=6.0)

    config = CollisionConfig(
        use_footprint=True,
        body_length=2.0,
        body_width=1.0,
        wheelbase=2.0,
        rear_axle_ratio=0.5,
        origin_at_rear_axle=True,
    )

    checker = CollisionChecker(track, config=config)

    s = _state(10.0, 0.0)
    assert not checker.is_collision(s)


# ================================================================
# Vectorized collision tests
# ================================================================


def test_vectorized_point_collision() -> None:
    """Vectorized point-based collision detection."""
    track = _straight_track(width=4.0)
    checker = CollisionChecker(track)

    states = StateArray(
        x=np.array([10.0, 10.0], dtype=np.float64),
        y=np.array([0.0, 3.0], dtype=np.float64),
        heading=np.zeros(2, dtype=np.float64),
        velocity=np.zeros(2, dtype=np.float64),
        steering_angle=np.zeros(2, dtype=np.float64),
        progress_s=np.zeros(2, dtype=np.float64),
    )

    result = checker.is_collision_array(states)

    assert result.dtype == np.bool_
    assert result.shape == (2,)
    assert not result[0]
    assert result[1]


def test_vectorized_footprint_collision() -> None:
    """Vectorized footprint-based collision detection."""
    track = _straight_track(width=4.0)

    config = CollisionConfig(
        use_footprint=True,
        body_length=4.0,
        body_width=2.0,
        origin_at_rear_axle=False,
    )

    checker = CollisionChecker(track, config=config)

    states = StateArray(
        x=np.array([10.0, 10.0], dtype=np.float64),
        y=np.array([0.0, 1.5], dtype=np.float64),
        heading=np.zeros(2, dtype=np.float64),
        velocity=np.zeros(2, dtype=np.float64),
        steering_angle=np.zeros(2, dtype=np.float64),
        progress_s=np.zeros(2, dtype=np.float64),
    )

    result = checker.is_collision_array(states)

    assert result.dtype == np.bool_
    assert result.shape == (2,)
    assert not result[0]
    assert result[1]
