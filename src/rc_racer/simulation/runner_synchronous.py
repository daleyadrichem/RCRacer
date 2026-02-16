"""
Synchronous (single-episode, synchronous) simulation runner.

SIMULATION Layer
----------------
This runner sits between batch and realtime execution:

- Runs exactly one simulation at a time
- Fully synchronous
- No wall-clock pacing
- No threading
- No multiprocessing
- Deterministic given seed
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
    Structural protocol for synchronous runner.

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
        Advance environment by one fixed dt.
        """


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class SynchronousRunnerConfig:
    """
    Configuration for synchronous execution.

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
class SynchronousEpisodeResult:
    """
    Summary of a completed synchronous episode.

    Parameters
    ----------
    steps : int
        Number of executed steps.
    total_reward : float
        Accumulated reward.
    terminated : bool
        True if environment terminated naturally.
    final_state : State
        Final environment state.
    """

    steps: int
    total_reward: float
    terminated: bool
    final_state: State


# ================================================================
# RUNNER
# ================================================================


class SynchronousRunner:
    """
    Authoritative synchronous simulation loop (single episode).

    Loop structure:

    while not done and steps < max_steps:
        action = controller.compute_action(state)
        state, reward, done = env.step(action)

    Notes
    -----
    - Environment controls dt.
    - No sleeping.
    - No real-time pacing.
    - No threading.
    - Deterministic given seed.
    - Intended for debugging, benchmarking, and controlled evaluation.
    """

    def __init__(
        self,
        env: EnvironmentLike,
        controller: BaseController,
        config: SynchronousRunnerConfig,
    ) -> None:
        """
        Initialize synchronous runner.

        Parameters
        ----------
        env : EnvironmentLike
            Deterministic environment instance.
        controller : BaseController
            Controller instance.
        config : SynchronousRunnerConfig
            Execution configuration.
        """
        self._env: EnvironmentLike = env
        self._controller: BaseController = controller
        self._config: SynchronousRunnerConfig = config

    # ------------------------------------------------------------

    @property
    def config(self) -> SynchronousRunnerConfig:
        """
        Return runner configuration.

        Returns
        -------
        SynchronousRunnerConfig
        """
        return self._config

    # ------------------------------------------------------------

    def run(
        self,
        *,
        seed: int | None = None,
    ) -> SynchronousEpisodeResult:
        """
        Execute a single deterministic episode.

        Parameters
        ----------
        seed : int | None
            Optional reset seed forwarded to environment.

        Returns
        -------
        SynchronousEpisodeResult
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

        return SynchronousEpisodeResult(
            steps=steps,
            total_reward=total_reward,
            terminated=bool(done),
            final_state=state,
        )
