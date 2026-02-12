"""
test_state.py

Comprehensive unit tests for rc_racer.core.state.

Covers:
- Scalar State
- Vectorized StateArray
- Validation logic
- Serialization
- Determinism
- Edge cases

All tests are deterministic and architecture-compliant.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rc_racer.core.state import State, StateArray


# ============================================================
# Scalar State Tests
# ============================================================


def test_state_initialization() -> None:
    state = State(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

    assert state.x == 1.0
    assert state.y == 2.0
    assert state.heading == 3.0
    assert state.velocity == 4.0
    assert state.steering_angle == 5.0
    assert state.progress_s == 6.0


def test_state_velocity_validation() -> None:
    with pytest.raises(ValueError):
        State(0.0, 0.0, 0.0, -1.0, 0.0, 0.0)


def test_state_is_immutable() -> None:
    state = State(0, 0, 0, 1, 0, 0)

    with pytest.raises(Exception):
        state.x = 10.0  # type: ignore


def test_state_as_tuple() -> None:
    state = State(1, 2, 3, 4, 5, 6)
    assert state.as_tuple() == (1, 2, 3, 4, 5, 6)


def test_state_copy_with_single_field() -> None:
    state = State(1, 2, 3, 4, 5, 6)
    new_state = state.copy_with(velocity=10.0)

    assert new_state.velocity == 10.0
    assert new_state.x == 1
    assert new_state.y == 2


def test_state_copy_with_multiple_fields() -> None:
    state = State(1, 2, 3, 4, 5, 6)
    new_state = state.copy_with(x=10, y=20, heading=0)

    assert new_state.x == 10
    assert new_state.y == 20
    assert new_state.heading == 0
    assert new_state.velocity == 4


def test_state_copy_with_no_changes() -> None:
    state = State(1, 2, 3, 4, 5, 6)
    new_state = state.copy_with()

    assert new_state == state
    assert new_state is not state


def test_state_equality_and_hash() -> None:
    s1 = State(1, 2, 3, 4, 5, 6)
    s2 = State(1, 2, 3, 4, 5, 6)

    assert s1 == s2
    assert hash(s1) == hash(s2)

    state_set = {s1}
    assert s2 in state_set


def test_state_dict_roundtrip() -> None:
    state = State(1, 2, 3, 4, 5, 6)

    data = state.to_dict()
    restored = State.from_dict(data)

    assert restored == state


def test_state_json_roundtrip() -> None:
    state = State(1, 2, 3, 4, 5, 6)

    json_str = json.dumps(state.to_dict())
    loaded = json.loads(json_str)

    restored = State.from_dict(loaded)
    assert restored == state


# ============================================================
# StateArray Tests
# ============================================================


def make_valid_array(n: int) -> StateArray:
    return StateArray(
        x=np.zeros(n, dtype=np.float64),
        y=np.zeros(n, dtype=np.float64),
        heading=np.zeros(n, dtype=np.float64),
        velocity=np.ones(n, dtype=np.float64),
        steering_angle=np.zeros(n, dtype=np.float64),
        progress_s=np.zeros(n, dtype=np.float64),
    )


def test_state_array_creation() -> None:
    arr = make_valid_array(5)
    assert arr.batch_size == 5


def test_state_array_zero_length() -> None:
    arr = make_valid_array(0)
    assert arr.batch_size == 0


def test_state_array_shape_validation() -> None:
    with pytest.raises(ValueError):
        StateArray(
            x=np.zeros(3),
            y=np.zeros(2),
            heading=np.zeros(3),
            velocity=np.ones(3),
            steering_angle=np.zeros(3),
            progress_s=np.zeros(3),
        )


def test_state_array_dtype_validation() -> None:
    with pytest.raises(ValueError):
        StateArray(
            x=np.zeros(2, dtype=np.float32),
            y=np.zeros(2, dtype=np.float64),
            heading=np.zeros(2, dtype=np.float64),
            velocity=np.ones(2, dtype=np.float64),
            steering_angle=np.zeros(2, dtype=np.float64),
            progress_s=np.zeros(2, dtype=np.float64),
        )


def test_state_array_velocity_validation() -> None:
    with pytest.raises(ValueError):
        StateArray(
            x=np.zeros(2),
            y=np.zeros(2),
            heading=np.zeros(2),
            velocity=np.array([1.0, -1.0]),
            steering_angle=np.zeros(2),
            progress_s=np.zeros(2),
        )


def test_state_array_dict_roundtrip() -> None:
    arr = make_valid_array(4)

    data = arr.to_dict()
    restored = StateArray.from_dict(data)

    assert np.array_equal(arr.x, restored.x)
    assert np.array_equal(arr.velocity, restored.velocity)


def test_state_array_json_roundtrip() -> None:
    arr = make_valid_array(3)

    json_str = json.dumps(arr.to_dict())
    loaded = json.loads(json_str)

    restored = StateArray.from_dict(loaded)

    assert np.array_equal(arr.heading, restored.heading)


def test_state_array_large_batch() -> None:
    n = 1000
    arr = make_valid_array(n)

    assert arr.batch_size == n
    assert arr.x.shape == (n,)
    assert arr.velocity.dtype == np.float64


def test_state_array_independent_instances() -> None:
    arr1 = make_valid_array(3)
    arr2 = make_valid_array(3)

    assert arr1 is not arr2
    assert np.array_equal(arr1.x, arr2.x)
