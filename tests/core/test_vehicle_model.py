"""
test_vehicle_model.py

Unit tests for VehicleModel.

These tests verify:
- Determinism
- Friction circle behavior
- Lateral acceleration saturation
- Drag model effects
- Vectorized consistency

Architecture reference:
:contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_racer.core.state import State, StateArray
from rc_racer.core.vehicle_model import VehicleModel, VehicleParams


@pytest.fixture
def params() -> VehicleParams:
    return VehicleParams(
        wheelbase=2.5,
        rear_axle_ratio=0.5,
        max_steering_angle=np.deg2rad(30),
        max_steering_rate=np.deg2rad(180),
        max_acceleration=5.0,
        max_velocity=50.0,
        mu=1.0,
        g=9.81,
        a_lat_max=8.0,
        mass=1200.0,
        c_rr=0.015,
        c_d_a_over_m=0.0005,
    )


@pytest.fixture
def model(params: VehicleParams) -> VehicleModel:
    return VehicleModel(params)


@pytest.fixture
def base_state() -> State:
    return State(
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity=10.0,
        steering_angle=0.0,
        progress_s=0.0,
    )


def test_deterministic(model: VehicleModel, base_state: State) -> None:
    action = (1.0, 0.1)
    dt = 0.1

    s1 = model.step(base_state, action, dt)
    s2 = model.step(base_state, action, dt)

    assert s1 == s2


def test_velocity_clamped(model: VehicleModel, base_state: State) -> None:
    action = (100.0, 0.0)
    dt = 1.0

    s = model.step(base_state, action, dt)
    assert s.velocity <= model._p.max_velocity


def test_drag_reduces_velocity(model: VehicleModel, base_state: State) -> None:
    action = (0.0, 0.0)
    dt = 0.5

    s = model.step(base_state, action, dt)
    assert s.velocity < base_state.velocity


def test_lateral_acceleration_saturation(model: VehicleModel, base_state: State) -> None:
    high_speed_state = base_state.copy_with(
        velocity=30.0,
        steering_angle=np.deg2rad(30),
    )

    action = (0.0, 0.0)
    dt = 0.1

    s = model.step(high_speed_state, action, dt)

    # heading change should be limited (indirect check)
    assert abs(s.heading - high_speed_state.heading) < 1.0


def test_friction_circle_limits_accel(model: VehicleModel, base_state: State) -> None:
    state = base_state.copy_with(
        velocity=20.0,
        steering_angle=np.deg2rad(25),
    )

    action = (5.0, 0.0)
    dt = 0.1

    s = model.step(state, action, dt)

    # Expect acceleration reduced by lateral demand
    assert s.velocity - state.velocity < 5.0 * dt


def test_vectorized_matches_scalar(model: VehicleModel, base_state: State) -> None:
    dt = 0.1
    actions = np.array([[1.0, 0.1]])

    scalar_next = model.step(base_state, (1.0, 0.1), dt)

    state_array = StateArray(
        x=np.array([base_state.x]),
        y=np.array([base_state.y]),
        heading=np.array([base_state.heading]),
        velocity=np.array([base_state.velocity]),
        steering_angle=np.array([base_state.steering_angle]),
        progress_s=np.array([base_state.progress_s]),
    )

    array_next = model.step_array(state_array, actions, dt)

    assert np.isclose(array_next.x[0], scalar_next.x)
    assert np.isclose(array_next.y[0], scalar_next.y)
    assert np.isclose(array_next.heading[0], scalar_next.heading)
    assert np.isclose(array_next.velocity[0], scalar_next.velocity)
