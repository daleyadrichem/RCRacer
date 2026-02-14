"""
rc_racer.simulation.runner_batch

Batch (headless, synchronous) simulation runner.

SIMULATION Layer
----------------
Used for:
- Reinforcement learning
- Evolutionary algorithms
- Large-scale evaluation
- Deterministic benchmarking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

from rc_racer.agents.base_controller import BaseController
from rc_racer.core.state import State


Action = Tuple[float, float]


# ================================================================
# ENVIRONMENT PROTOCOL
# ================================================================


class EnvironmentLike(Protocol):
    """
    Structural protocol for batch runner.

    The environment must expose a Gym-like synchronous API.
    """

    @property
    def state(self) -> State:
        """
        Current immutable state.
        """

    def reset(self, seed: int | None = None) -> State:
        """
        Reset environment.
        """

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        """
        Step environment forward by one fixed dt.
        """


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class BatchRunnerConfig:
    """
    Configuration for batch execution.

    Parameters
    ----------
    max_steps : int
        Maximum number of environment steps.
    """

    max_steps: int


# ================================================================
# RESULT CONTAINER
# ================================================================


@dataclass(frozen=True)
class BatchEpisodeResult:
    """
    Summary of a completed batch episode.

    Parameters
    ----------
    steps : int
        Number of executed steps.
    total_reward : float
        Accumulated reward.
    terminated : bool
        True if environment terminated naturally.
    """

    steps: int
    total_reward: float
    terminated: bool


# ================================================================
# RUNNER
# ================================================================


class BatchRunner:
    """
    Authoritative synchronous simulation loop.

    Loop structure:

    for t in range(max_steps):
        action = controller.compute_action(state)
        state, reward, done = env.step(action)

    Notes
    -----
    - Environment controls dt.
    - No sleeping.
    - No real-time logic.
    - Safe for multiprocessing.
    - Deterministic given seed.
    """

    def __init__(
        self,
        env: EnvironmentLike,
        controller: BaseController,
        config: BatchRunnerConfig,
    ) -> None:
        """
        Initialize batch runner.

        Parameters
        ----------
        env : EnvironmentLike
            Deterministic environment.
        controller : BaseController
            Controller instance.
        config : BatchRunnerConfig
            Execution configuration.
        """
        self._env: EnvironmentLike = env
        self._controller: BaseController = controller
        self._config: BatchRunnerConfig = config

    # ------------------------------------------------------------

    @property
    def config(self) -> BatchRunnerConfig:
        """
        Return runner configuration.

        Returns
        -------
        BatchRunnerConfig
        """
        return self._config

    # ------------------------------------------------------------

    def run(
        self,
        *,
        seed: int | None = None,
    ) -> BatchEpisodeResult:
        """
        Execute a single deterministic episode.

        Parameters
        ----------
        seed : int | None
            Optional reset seed forwarded to environment.

        Returns
        -------
        BatchEpisodeResult
            Episode summary.
        """
        state: State = self._env.reset(seed=seed)
        self._controller.reset()

        total_reward: float = 0.0
        steps: int = 0
        done: bool = False

        while not done and steps < self._config.max_steps:
            action: Action = self._controller.compute_action(state)

            next_state, reward, done, _info = self._env.step(action)

            total_reward += float(reward)
            steps += 1
            state = next_state

        return BatchEpisodeResult(
            steps=steps,
            total_reward=total_reward,
            terminated=bool(done),
        )
