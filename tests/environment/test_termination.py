"""
Unit tests for the TerminationCondition logic.

Tests cover:
- Collision termination
- Lap completion
- Timeout
- Reverse driving detection
- Reverse allowed configuration
- Epsilon tolerance behavior
- Step counter tracking

This module tests:
    rc_racer.environment.termination

The implementation under test is defined in:
    termination.py
"""

from __future__ import annotations

import pytest

from rc_racer.core.state import State
from rc_racer.environment.termination import (
    TerminationCondition,
    TerminationConfig,
)


# ============================================================
# Helpers
# ============================================================


def make_state(progress: float) -> State:
    """
    Create a minimal valid state for testing.

    Parameters
    ----------
    progress : float
        Arc-length progress value.

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


# ============================================================
# Collision
# ============================================================


def test_collision_triggers_termination() -> None:
    """
    Episode must terminate immediately when collision=True.
    """
    config = TerminationConfig(max_steps=100)
    term = TerminationCondition(total_track_length=100.0, config=config)

    initial = make_state(progress=0.0)
    term.reset(initial)

    done = term.check(make_state(progress=1.0), collision=True)

    assert done is True


# ============================================================
# Lap Completion
# ============================================================


def test_lap_completion_triggers_termination() -> None:
    """
    Episode must terminate when progress >= total_track_length.
    """
    config = TerminationConfig(max_steps=100)
    term = TerminationCondition(total_track_length=50.0, config=config)

    initial = make_state(progress=0.0)
    term.reset(initial)

    done = term.check(make_state(progress=50.0), collision=False)

    assert done is True


# ============================================================
# Timeout
# ============================================================


def test_timeout_triggers_termination() -> None:
    """
    Episode must terminate when step_count >= max_steps.
    """
    config = TerminationConfig(max_steps=3)
    term = TerminationCondition(total_track_length=100.0, config=config)

    initial = make_state(progress=0.0)
    term.reset(initial)

    # Step 1
    assert term.check(make_state(progress=1.0), collision=False) is False
    # Step 2
    assert term.check(make_state(progress=2.0), collision=False) is False
    # Step 3 → should terminate
    assert term.check(make_state(progress=3.0), collision=False) is True


# ============================================================
# Reverse Driving
# ============================================================


def test_reverse_progress_triggers_termination_when_not_allowed() -> None:
    """
    If allow_reverse=False and progress decreases beyond epsilon,
    termination must occur.
    """
    config = TerminationConfig(max_steps=100, allow_reverse=False)
    term = TerminationCondition(total_track_length=100.0, config=config)

    initial = make_state(progress=10.0)
    term.reset(initial)

    done = term.check(make_state(progress=9.0), collision=False)

    assert done is True


def test_reverse_progress_allowed_when_configured() -> None:
    """
    If allow_reverse=True, decreasing progress should NOT terminate.
    """
    config = TerminationConfig(max_steps=100, allow_reverse=True)
    term = TerminationCondition(total_track_length=100.0, config=config)

    initial = make_state(progress=10.0)
    term.reset(initial)

    done = term.check(make_state(progress=9.0), collision=False)

    assert done is False


# ============================================================
# Epsilon Tolerance
# ============================================================


def test_reverse_within_epsilon_does_not_terminate() -> None:
    """
    Small numerical decrease within epsilon must not trigger termination.
    """
    config = TerminationConfig(
        max_steps=100,
        allow_reverse=False,
        progress_epsilon=1e-3,
    )
    term = TerminationCondition(total_track_length=100.0, config=config)

    initial = make_state(progress=10.0)
    term.reset(initial)

    # Decrease smaller than epsilon
    done = term.check(make_state(progress=10.0 - 5e-4), collision=False)

    assert done is False


# ============================================================
# Step Counter
# ============================================================


def test_step_count_increments_correctly() -> None:
    """
    step_count property must reflect number of check calls.
    """
    config = TerminationConfig(max_steps=100)
    term = TerminationCondition(total_track_length=100.0, config=config)

    initial = make_state(progress=0.0)
    term.reset(initial)

    assert term.step_count == 0

    term.check(make_state(progress=1.0), collision=False)
    assert term.step_count == 1

    term.check(make_state(progress=2.0), collision=False)
    assert term.step_count == 2
