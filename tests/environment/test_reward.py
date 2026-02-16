"""
Unit tests for rc_racer.environment.reward.

These tests validate the deterministic and stateless behavior of
RewardSystem as defined in the Environment layer.

Test Coverage
-------------
- Progress reward
- Time penalty
- Off-track penalty
- Finish bonus
- Combined reward composition
- Reverse progress handling
- Determinism
"""

from __future__ import annotations

import math

import pytest

from rc_racer.core.state import State
from rc_racer.environment.reward import RewardConfig, RewardSystem


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def reward_system() -> RewardSystem:
    """
    Create a standard reward system for testing.

    Returns
    -------
    RewardSystem
    """
    config = RewardConfig(
        progress_weight=2.0,
        off_track_penalty=5.0,
        time_penalty=0.1,
        finish_bonus=50.0,
    )
    return RewardSystem(config)


def _make_state(progress: float) -> State:
    """
    Create a minimal valid State with specified progress.

    Parameters
    ----------
    progress : float

    Returns
    -------
    State
    """
    return State(
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity=1.0,
        steering_angle=0.0,
        progress_s=progress,
    )


# ================================================================
# Tests
# ================================================================


def test_progress_reward_only(reward_system: RewardSystem) -> None:
    """
    Reward should equal progress_weight * delta_progress
    minus time penalty.
    """
    prev = _make_state(1.0)
    curr = _make_state(2.5)

    reward = reward_system.compute(
        prev,
        curr,
        is_off_track=False,
        lap_completed=False,
    )

    expected = 2.0 * (1.5) - 0.1
    assert math.isclose(reward, expected, rel_tol=1e-12)


def test_time_penalty_always_applied(reward_system: RewardSystem) -> None:
    """
    Time penalty should apply even if no progress.
    """
    prev = _make_state(1.0)
    curr = _make_state(1.0)

    reward = reward_system.compute(
        prev,
        curr,
        is_off_track=False,
        lap_completed=False,
    )

    expected = -0.1
    assert math.isclose(reward, expected, rel_tol=1e-12)


def test_off_track_penalty(reward_system: RewardSystem) -> None:
    """
    Off-track penalty should subtract configured value.
    """
    prev = _make_state(0.0)
    curr = _make_state(1.0)

    reward = reward_system.compute(
        prev,
        curr,
        is_off_track=True,
        lap_completed=False,
    )

    expected = 2.0 * 1.0 - 0.1 - 5.0
    assert math.isclose(reward, expected, rel_tol=1e-12)


def test_finish_bonus(reward_system: RewardSystem) -> None:
    """
    Finish bonus should be added when lap_completed=True.
    """
    prev = _make_state(9.0)
    curr = _make_state(10.0)

    reward = reward_system.compute(
        prev,
        curr,
        is_off_track=False,
        lap_completed=True,
    )

    expected = 2.0 * 1.0 - 0.1 + 50.0
    assert math.isclose(reward, expected, rel_tol=1e-12)


def test_combined_reward_components(reward_system: RewardSystem) -> None:
    """
    All components should combine linearly and deterministically.
    """
    prev = _make_state(2.0)
    curr = _make_state(4.0)

    reward = reward_system.compute(
        prev,
        curr,
        is_off_track=True,
        lap_completed=True,
    )

    expected = (
        2.0 * 2.0  # progress
        - 0.1      # time penalty
        - 5.0      # off-track
        + 50.0     # finish bonus
    )

    assert math.isclose(reward, expected, rel_tol=1e-12)


def test_negative_progress_penalized(reward_system: RewardSystem) -> None:
    """
    Negative progress should produce negative reward.
    """
    prev = _make_state(5.0)
    curr = _make_state(4.0)

    reward = reward_system.compute(
        prev,
        curr,
        is_off_track=False,
        lap_completed=False,
    )

    expected = 2.0 * (-1.0) - 0.1
    assert math.isclose(reward, expected, rel_tol=1e-12)


def test_stateless_determinism(reward_system: RewardSystem) -> None:
    """
    Reward computation must be deterministic and stateless.
    """
    prev = _make_state(3.0)
    curr = _make_state(5.0)

    r1 = reward_system.compute(
        prev,
        curr,
        is_off_track=False,
        lap_completed=False,
    )

    r2 = reward_system.compute(
        prev,
        curr,
        is_off_track=False,
        lap_completed=False,
    )

    assert r1 == r2
