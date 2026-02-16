"""
Unit tests for rc_racer.environment.collision.

These tests validate the public API contract of the CollisionChecker
skeleton implementation.

The current module is a structural skeleton, so collision methods are
expected to raise NotImplementedError.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from rc_racer.environment.collision import CollisionChecker, CollisionConfig


# ---------------------------------------------------------------------
# Dummy Test Stubs
# ---------------------------------------------------------------------


class DummyTrack:
    """Minimal Track stub for unit testing."""

    width: float = 10.0


class DummyState:
    """Minimal State stub for unit testing."""

    def __init__(
        self,
        x: float,
        y: float,
        heading: float = 0.0,
        velocity: float = 0.0,
        steering_angle: float = 0.0,
        progress_s: float = 0.0,
    ) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity
        self.steering_angle = steering_angle
        self.progress_s = progress_s


class DummyStateArray:
    """Minimal StateArray stub for unit testing."""

    def __init__(self, size: int) -> None:
        self.batch_size: int = size
        self.x: NDArray[np.float64] = np.zeros(size)
        self.y: NDArray[np.float64] = np.zeros(size)
        self.heading: NDArray[np.float64] = np.zeros(size)
        self.velocity: NDArray[np.float64] = np.zeros(size)
        self.steering_angle: NDArray[np.float64] = np.zeros(size)
        self.progress_s: NDArray[np.float64] = np.zeros(size)


# ---------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------


def test_collision_config_defaults() -> None:
    """Test default configuration values."""
    config = CollisionConfig()

    assert config.use_footprint is False
    assert config.body_length == 0.0
    assert config.body_width == 0.0
    assert config.margin == 0.0


# ---------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------


def test_collision_checker_initialization() -> None:
    """Ensure CollisionChecker initializes correctly."""
    track = DummyTrack()
    checker = CollisionChecker(track=track)

    assert checker.track is track
    assert isinstance(checker.config, CollisionConfig)


# ---------------------------------------------------------------------
# Skeleton Behavior Tests
# ---------------------------------------------------------------------


def test_is_collision_raises_not_implemented() -> None:
    """Skeleton should raise NotImplementedError."""
    track = DummyTrack()
    checker = CollisionChecker(track=track)
    state = DummyState(x=0.0, y=0.0)

    with pytest.raises(NotImplementedError):
        checker.is_collision(state)  # type: ignore[arg-type]


def test_is_collision_array_raises_not_implemented() -> None:
    """Skeleton should raise NotImplementedError for batch method."""
    track = DummyTrack()
    checker = CollisionChecker(track=track)
    states = DummyStateArray(size=5)

    with pytest.raises(NotImplementedError):
        checker.is_collision_array(states)  # type: ignore[arg-type]


def test_is_point_inside_track_raises_not_implemented() -> None:
    """Private method should raise NotImplementedError."""
    track = DummyTrack()
    checker = CollisionChecker(track=track)

    with pytest.raises(NotImplementedError):
        checker._is_point_inside_track(0.0, 0.0)


def test_footprint_corners_raises_not_implemented() -> None:
    """Private method should raise NotImplementedError."""
    track = DummyTrack()
    checker = CollisionChecker(track=track)
    state = DummyState(x=0.0, y=0.0)

    with pytest.raises(NotImplementedError):
        checker._footprint_corners(state)  # type: ignore[arg-type]
