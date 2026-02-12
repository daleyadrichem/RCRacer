"""
test_track.py

Unit tests for the Track class in core.track.

These tests validate:

- Arc-length parametrization
- Boundary computation
- Projection correctness
- Inside/outside detection
- Immutability constraints

All tests are deterministic and contain no randomness.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from rc_racer.core.track import Track


FloatArray = NDArray[np.float64]


def create_straight_track(length: float = 10.0, width: float = 2.0) -> Track:
    """
    Create a simple straight track along the x-axis.

    Parameters
    ----------
    length : float
        Track length.
    width : float
        Track width.

    Returns
    -------
    Track
        Constructed Track instance.
    """
    centerline: FloatArray = np.array(
        [[0.0, 0.0], [length, 0.0]],
        dtype=np.float64,
    )
    return Track(centerline=centerline, width=width)


def test_arc_length_computation() -> None:
    """
    Test that arc-length is computed correctly for a straight line.
    """
    track = create_straight_track(length=10.0)

    assert track.total_length == pytest.approx(10.0)
    assert track.arc_lengths.shape == (2,)
    assert track.arc_lengths[0] == 0.0
    assert track.arc_lengths[1] == pytest.approx(10.0)


def test_boundaries_parallel_to_centerline() -> None:
    """
    Boundaries of a straight horizontal track should be parallel
    and offset in y-direction.
    """
    track = create_straight_track(length=10.0, width=2.0)

    left = track.left_boundary
    right = track.right_boundary

    # Left boundary should be +1 in y
    assert np.allclose(left[:, 1], 1.0)

    # Right boundary should be -1 in y
    assert np.allclose(right[:, 1], -1.0)


def test_projection_on_centerline() -> None:
    """
    A point directly above the centerline should project vertically.
    """
    track = create_straight_track(length=10.0)

    point: FloatArray = np.array([5.0, 1.0], dtype=np.float64)

    s, projected = track.project(point)

    assert s == pytest.approx(5.0)
    assert np.allclose(projected, np.array([5.0, 0.0]))


def test_projection_clamps_to_segment() -> None:
    """
    Projection before start should clamp to first point.
    """
    track = create_straight_track(length=10.0)

    point: FloatArray = np.array([-5.0, 0.0], dtype=np.float64)

    s, projected = track.project(point)

    assert s == pytest.approx(0.0)
    assert np.allclose(projected, np.array([0.0, 0.0]))


def test_inside_track() -> None:
    """
    Points within half-width should be inside.
    """
    track = create_straight_track(length=10.0, width=2.0)

    inside_point: FloatArray = np.array([5.0, 0.5], dtype=np.float64)

    assert track.is_inside(inside_point)


def test_outside_track() -> None:
    """
    Points outside half-width should be detected.
    """
    track = create_straight_track(length=10.0, width=2.0)

    outside_point: FloatArray = np.array([5.0, 2.0], dtype=np.float64)

    assert not track.is_inside(outside_point)


def test_invalid_centerline_shape() -> None:
    """
    Invalid centerline shape should raise ValueError.
    """
    with pytest.raises(ValueError):
        Track(centerline=np.array([1.0, 2.0]), width=2.0)  # type: ignore[arg-type]


def test_invalid_width() -> None:
    """
    Non-positive width should raise ValueError.
    """
    centerline: FloatArray = np.array([[0.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError):
        Track(centerline=centerline, width=0.0)


def test_track_immutable() -> None:
    """
    Track should be frozen (immutable dataclass).
    """
    track = create_straight_track()

    with pytest.raises(Exception):
        track.width = 5.0  # type: ignore[misc]
