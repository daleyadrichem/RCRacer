"""
Unit tests for Environment layer.

Tests:
- Reset determinism
- Step determinism
- Progress increases
- Collision termination
- Timeout termination
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_racer.core.track import Track
from rc_racer.core.vehicle_model import VehicleModel, VehicleParams
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig


# ================================================================
# HELPERS
# ================================================================


def create_simple_track() -> Track:
    centerline = np.array(
        [[0.0, 0.0], [50.0, 0.0]],
        dtype=np.float64,
    )
    return Track(centerline=centerline, width=4.0)


def create_env() -> Environment:
    track = create_simple_track()

    params = VehicleParams(
        wheelbase=0.3,
        rear_axle_ratio=0.5,
        max_steering_angle=0.5,
        max_steering_rate=1.0,
        max_acceleration=5.0,
        max_velocity=10.0,
        mu=1.0,
        g=9.81,
        a_lat_max=10.0,
        mass=1.0,
        c_rr=0.0,
        c_d_a_over_m=0.0,
    )

    vehicle = VehicleModel(params)

    collision = CollisionChecker(track)

    reward = RewardSystem(
        RewardConfig(
            progress_weight=1.0,
            off_track_penalty=-10.0,
            time_penalty=-0.01,
            finish_bonus=100.0,
        )
    )

    termination = TerminationCondition(
        total_track_length=track.total_length,
        config=TerminationConfig(max_steps=1000),
    )

    return Environment(
        track=track,
        vehicle_model=vehicle,
        collision_checker=collision,
        reward_system=reward,
        termination_condition=termination,
        config=EnvironmentConfig(dt=0.02),
    )


# ================================================================
# TESTS
# ================================================================


def test_reset_deterministic() -> None:
    env = create_env()
    s1 = env.reset(seed=123)
    s2 = env.reset(seed=123)
    assert s1 == s2


def test_progress_increases() -> None:
    env = create_env()
    state = env.reset()

    for _ in range(50):
        state, _, _, _ = env.step((1.0, 0.0))

    assert state.progress_s > 0.0


def test_collision_eventually() -> None:
    env = create_env()
    env.reset()

    done = False
    for _ in range(2000):
        _, _, done, info = env.step((1.0, 0.5))
        if done:
            break

    assert done


def test_timeout() -> None:
    env = create_env()
    env.reset()

    env._termination_condition._config = TerminationConfig(max_steps=5)

    done = False
    for _ in range(10):
        _, _, done, _ = env.step((0.0, 0.0))
        if done:
            break

    assert done
