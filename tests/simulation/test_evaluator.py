"""
test_evaluator.py

Pytest tests for rc_racer.simulation.evaluator.Evaluator.

These tests verify:

- Deterministic behavior
- Correct seed handling
- Proper environment recreation
- Fitness equals total accumulated reward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pytest

from rc_racer.simulation.evaluator import Evaluator, EvaluatorConfig
from rc_racer.simulation.runner_batch import BatchRunnerConfig
from rc_racer.core.state import State


# ============================================================
# Dummy Test Components
# ============================================================


class DummyEnvironment:
    """
    Minimal deterministic environment for Evaluator testing.

    Behavior:
    - Reward = +1.0 per step
    - Terminates when internal counter reaches max_steps
    - Tracks reset seed for verification
    """

    def __init__(self, max_steps: int) -> None:
        self._max_steps: int = max_steps
        self._step_count: int = 0
        self.last_seed: int | None = None

    @property
    def state(self) -> State:
        return State(
            x=0.0,
            y=0.0,
            heading=0.0,
            velocity=0.0,
            steering_angle=0.0,
            progress_s=float(self._step_count),
        )

    def reset(self, seed: int | None = None) -> State:
        self._step_count = 0
        self.last_seed = seed
        return self.state

    def step(
        self,
        action: Tuple[float, float],
    ) -> tuple[State, float, bool, dict]:
        self._step_count += 1
        done = self._step_count >= self._max_steps
        reward = 1.0
        return self.state, reward, done, {}


class DummyController:
    """
    Minimal deterministic controller.
    """

    def __init__(self, genome: float) -> None:
        self.genome: float = genome
        self.reset_called: bool = False

    def reset(self) -> None:
        self.reset_called = True

    def compute_action(self, state: State) -> Tuple[float, float]:
        return (self.genome, 0.0)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def max_steps() -> int:
    return 5


@pytest.fixture
def env_factory(max_steps: int):
    instances = []

    def factory() -> DummyEnvironment:
        env = DummyEnvironment(max_steps=max_steps)
        instances.append(env)
        return env

    factory.instances = instances  # type: ignore[attr-defined]
    return factory


@pytest.fixture
def controller_factory():
    def factory(genome: float) -> DummyController:
        return DummyController(genome)

    return factory


# ============================================================
# Tests
# ============================================================


def test_evaluator_returns_total_reward(
    env_factory,
    controller_factory,
    max_steps: int,
) -> None:
    """
    Fitness should equal accumulated reward.
    """

    evaluator = Evaluator(
        env_factory=env_factory,
        controller_factory=controller_factory,
        config=EvaluatorConfig(max_steps=max_steps),
    )

    fitness = evaluator.evaluate(genome=1.0, seed=42)

    assert fitness == pytest.approx(float(max_steps))


def test_evaluator_respects_max_steps(
    env_factory,
    controller_factory,
) -> None:
    """
    Evaluator must pass max_steps into BatchRunner.
    """

    evaluator = Evaluator(
        env_factory=env_factory,
        controller_factory=controller_factory,
        config=EvaluatorConfig(max_steps=3),
    )

    fitness = evaluator.evaluate(genome=0.0)

    assert fitness == pytest.approx(3.0)


def test_seed_offset_is_applied(
    env_factory,
    controller_factory,
) -> None:
    """
    Seed passed to environment must include seed_offset.
    """

    evaluator = Evaluator(
        env_factory=env_factory,
        controller_factory=controller_factory,
        config=EvaluatorConfig(max_steps=2, seed_offset=10),
    )

    evaluator.evaluate(genome=0.0, seed=5)

    env_instance = env_factory.instances[-1]
    assert env_instance.last_seed == 15


def test_none_seed_remains_none(
    env_factory,
    controller_factory,
) -> None:
    """
    If seed=None, environment must receive None.
    """

    evaluator = Evaluator(
        env_factory=env_factory,
        controller_factory=controller_factory,
        config=EvaluatorConfig(max_steps=2, seed_offset=100),
    )

    evaluator.evaluate(genome=0.0, seed=None)

    env_instance = env_factory.instances[-1]
    assert env_instance.last_seed is None


def test_environment_is_recreated_each_evaluation(
    env_factory,
    controller_factory,
) -> None:
    """
    Evaluator must create a fresh environment per call.
    """

    evaluator = Evaluator(
        env_factory=env_factory,
        controller_factory=controller_factory,
        config=EvaluatorConfig(max_steps=2),
    )

    evaluator.evaluate(genome=0.0)
    evaluator.evaluate(genome=0.0)

    assert len(env_factory.instances) == 2
    assert env_factory.instances[0] is not env_factory.instances[1]


def test_evaluator_config_property(
    env_factory,
    controller_factory,
) -> None:
    """
    config property should return the original EvaluatorConfig.
    """

    config = EvaluatorConfig(max_steps=7, seed_offset=3)

    evaluator = Evaluator(
        env_factory=env_factory,
        controller_factory=controller_factory,
        config=config,
    )

    assert evaluator.config is config
