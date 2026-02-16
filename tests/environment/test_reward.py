"""
Unit tests for RewardSystem skeleton.

These tests validate:

- Correct configuration structure
- Proper initialization
- API contract
- Deterministic/stateless design expectations
- That compute() is intentionally unimplemented
"""

from __future__ import annotations

import pytest

from rc_racer.environment.reward import RewardConfig, RewardSystem
from rc_racer.core.state import State


# ================================================================
# FIXTURES
# ================================================================


@pytest.fixture()
def reward_config() -> RewardConfig:
    """
    Standard test configuration.
    """
    return RewardConfig(
        progress_weight=1.0,
        off_track_penalty=5.0,
        time_penalty=0.1,
        finish_bonus=100.0,
    )


@pytest.fixture()
def reward_system(reward_config: RewardConfig) -> RewardSystem:
    """
    Reward system instance.
    """
    return RewardSystem(reward_config)


@pytest.fixture()
def state_pair() -> tuple[State, State]:
    """
    Create two valid State instances.
    """
    s1: State = State(
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity=1.0,
        steering_angle=0.0,
        progress_s=10.0,
    )

    s2: State = s1.copy_with(progress_s=12.0)

    return s1, s2


# ================================================================
# CONFIG TESTS
# ================================================================


def test_reward_config_is_frozen(reward_config: RewardConfig) -> None:
    """
    RewardConfig must be immutable.
    """
    with pytest.raises(Exception):
        reward_config.progress_weight = 2.0  # type: ignore[attr-defined]


def test_reward_system_stores_config(
    reward_system: RewardSystem,
    reward_config: RewardConfig,
) -> None:
    """
    RewardSystem must store configuration.
    """
    assert reward_system._config == reward_config  # intentional internal check


# ================================================================
# API CONTRACT TESTS
# ================================================================


def test_compute_signature_exists(
    reward_system: RewardSystem,
    state_pair: tuple[State, State],
) -> None:
    """
    compute() must exist and accept required parameters.
    """
    s1, s2 = state_pair

    with pytest.raises(NotImplementedError):
        reward_system.compute(
            s1,
            s2,
            is_off_track=False,
            lap_completed=False,
        )


def test_compute_is_stateless(
    reward_system: RewardSystem,
    state_pair: tuple[State, State],
) -> None:
    """
    Multiple calls must not alter system state.

    For skeleton, we ensure consistent NotImplementedError.
    """
    s1, s2 = state_pair

    with pytest.raises(NotImplementedError):
        reward_system.compute(
            s1,
            s2,
            is_off_track=False,
            lap_completed=False,
        )

    with pytest.raises(NotImplementedError):
        reward_system.compute(
            s1,
            s2,
            is_off_track=False,
            lap_completed=False,
        )


# ================================================================
# STATE IMMUTABILITY GUARD
# ================================================================


def test_state_not_modified_on_compute_attempt(
    reward_system: RewardSystem,
    state_pair: tuple[State, State],
) -> None:
    """
    Even if compute fails, input states must remain unchanged.
    """
    s1, s2 = state_pair

    original_progress_1: float = s1.progress_s
    original_progress_2: float = s2.progress_s

    with pytest.raises(NotImplementedError):
        reward_system.compute(
            s1,
            s2,
            is_off_track=False,
            lap_completed=False,
        )

    assert s1.progress_s == original_progress_1
    assert s2.progress_s == original_progress_2
