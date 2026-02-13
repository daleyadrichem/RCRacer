"""
rc_racer.simulation.runner_realtime

Realtime (wall-clock paced) simulation runner.

The environment is stepped with a fixed timestep ``dt``. Sleeping is used only
to pace wall-clock speed and never affects simulation determinism.

This file follows the Simulation-layer rules:
- The simulation loop is authoritative (fixed dt, deterministic stepping).
- Controllers may be synchronous or run asynchronously (thread/process), but the
  environment remains single-threaded and synchronous. fileciteturn0file0
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import queue
import threading
import time
from typing import Callable, Optional, Protocol, Tuple

from rc_racer.agents.base_controller import BaseController
from rc_racer.core.state import State

Action = Tuple[float, float]


class EnvironmentLike(Protocol):
    """Structural protocol for realtime runners.

    Notes
    -----
    This runner intentionally depends only on a minimal environment surface.
    """

    @property
    def state(self) -> State:  # pragma: no cover
        """Current immutable state snapshot."""

    def reset(self, seed: int | None = None) -> State:  # pragma: no cover
        """Reset the episode."""

    def step(self, action: Action) -> tuple[State, float, bool, dict[str, float]]:  # pragma: no cover
        """Advance by one fixed timestep."""


class ActionProvider(Protocol):
    """Action source for the realtime loop."""

    def reset(self) -> None:  # pragma: no cover
        """Reset any internal state."""

    def get_action(self, state: State) -> Action:  # pragma: no cover
        """Return the action to apply for the current timestep."""


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for realtime execution.

    Parameters
    ----------
    dt : float
        Fixed simulation timestep in seconds.
    target_fps : float | None
        Wall-clock pacing target. If None, do not sleep (run as fast as possible).
    max_steps : int | None
        Optional hard cap on environment steps for the run.
    busy_wait : bool
        If True, use a short busy-wait for sub-millisecond accuracy.
    """

    dt: float
    target_fps: float | None = 60.0
    max_steps: int | None = None
    busy_wait: bool = False


@dataclass(frozen=True)
class EpisodeResult:
    """Result summary returned by :meth:`RealtimeRunner.run`.

    Parameters
    ----------
    steps : int
        Number of environment steps executed.
    total_reward : float
        Sum of per-step rewards.
    wall_time_s : float
        Total wall-clock time.
    achieved_fps : float
        Observed wall-clock loop frequency.
    """

    steps: int
    total_reward: float
    wall_time_s: float
    achieved_fps: float


class SyncControllerProvider(ActionProvider):
    """Synchronous provider that calls the controller directly."""

    def __init__(self, controller: BaseController) -> None:
        self._controller: BaseController = controller

    def reset(self) -> None:
        self._controller.reset()

    def get_action(self, state: State) -> Action:
        return self._controller.get_latest_action(state)


class QueueActionProvider(ActionProvider):
    """Provider that consumes the latest action from a queue.

    The queue is expected to contain actions of type ``Action``.
    If the queue is empty, the previous action is re-used.

    Parameters
    ----------
    action_queue : queue.Queue[Action] | mp.queues.Queue
        Queue supplying actions.
    default_action : tuple[float, float]
        Initial action to use before any actions arrive.
    """

    def __init__(
        self,
        action_queue: "queue.Queue[Action] | mp.queues.Queue",
        default_action: Action = (0.0, 0.0),
    ) -> None:
        self._queue = action_queue
        self._latest: Action = default_action

    def reset(self) -> None:
        self._latest = (0.0, 0.0)
        while True:
            try:
                _ = self._queue.get_nowait()
            except Exception:
                break

    def get_action(self, state: State) -> Action:
        del state
        while True:
            try:
                self._latest = self._queue.get_nowait()
            except Exception:
                break
        return self._latest


class ControllerWorkerThread:
    """Continuously compute actions in a background thread.

    The worker reads the latest state snapshot provided by the runner and
    pushes actions onto a queue. The environment is never accessed by the
    worker.

    Parameters
    ----------
    controller : BaseController
        Controller instance.
    action_queue : queue.Queue[Action]
        Queue to publish actions.
    poll_sleep_s : float
        Sleep duration when no state is available.
    """

    def __init__(
        self,
        controller: BaseController,
        action_queue: "queue.Queue[Action]",
        *,
        poll_sleep_s: float = 0.0,
    ) -> None:
        self._controller = controller
        self._action_queue = action_queue
        self._poll_sleep_s = float(poll_sleep_s)

        self._state_lock = threading.Lock()
        self._latest_state: State | None = None

        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start the worker thread."""
        self._controller.reset()
        self._thread.start()

    def stop(self) -> None:
        """Request the worker to stop and join it."""
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def push_state(self, state: State) -> None:
        """Update the latest state snapshot for action computation."""
        with self._state_lock:
            self._latest_state = state

    def _run(self) -> None:
        """Thread loop."""
        while not self._stop_evt.is_set():
            with self._state_lock:
                state = self._latest_state

            if state is None:
                if self._poll_sleep_s > 0.0:
                    time.sleep(self._poll_sleep_s)
                continue

            action = self._controller.compute_action(state)
            try:
                self._action_queue.put_nowait(action)
            except queue.Full:
                # Best-effort: keep the most recent action.
                try:
                    _ = self._action_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._action_queue.put_nowait(action)
                except queue.Full:
                    pass


class RealtimeRunner:
    """Authoritative realtime simulation runner.

    Parameters
    ----------
    env : EnvironmentLike
        Environment instance (single-threaded, synchronous).
    action_provider : ActionProvider
        Source of actions. May be synchronous or queue-backed.
    config : RunnerConfig
        Realtime pacing + step configuration.
    on_step : Callable[[State, float, bool, dict[str, float]], None] | None
        Optional callback invoked after every environment step. Use this to push
        state snapshots to a GUI without letting the GUI affect the simulation.
    """

    def __init__(
        self,
        env: EnvironmentLike,
        action_provider: ActionProvider,
        config: RunnerConfig,
        *,
        on_step: Callable[[State, float, bool, dict[str, float]], None] | None = None,
    ) -> None:
        self._env = env
        self._provider = action_provider
        self._config = config
        self._on_step = on_step

        self._stop_flag = False

    @property
    def config(self) -> RunnerConfig:
        """Return the runner configuration."""
        return self._config

    def stop(self) -> None:
        """Request the run loop to stop at the next iteration boundary."""
        self._stop_flag = True

    def run(self, *, seed: int | None = None) -> EpisodeResult:
        """Run a single realtime-paced episode.

        Parameters
        ----------
        seed : int | None
            Optional reset seed forwarded to the environment.

        Returns
        -------
        EpisodeResult
            Run summary.
        """
        self._stop_flag = False

        _ = self._env.reset(seed=seed)
        self._provider.reset()

        target_fps = self._config.target_fps
        target_period_s = (1.0 / float(target_fps)) if target_fps is not None else 0.0

        steps = 0
        total_reward = 0.0
        done = False

        t_start = time.monotonic()
        t_next = t_start

        while not done and not self._stop_flag:
            if self._config.max_steps is not None and steps >= self._config.max_steps:
                break

            state = self._env.state
            action = self._provider.get_action(state)

            next_state, reward, done, info = self._env.step(action)
            steps += 1
            total_reward += float(reward)

            if self._on_step is not None:
                self._on_step(next_state, float(reward), bool(done), info)

            if target_fps is not None:
                t_next += target_period_s
                self._sleep_until(t_next, busy_wait=self._config.busy_wait)

        wall_time_s = max(1e-12, time.monotonic() - t_start)
        achieved_fps = float(steps) / wall_time_s

        return EpisodeResult(
            steps=steps,
            total_reward=total_reward,
            wall_time_s=wall_time_s,
            achieved_fps=achieved_fps,
        )

    @staticmethod
    def _sleep_until(target_time: float, *, busy_wait: bool) -> None:
        """Sleep until the given monotonic time.

        Parameters
        ----------
        target_time : float
            Absolute target time (``time.monotonic()`` basis).
        busy_wait : bool
            If True, finish with a short busy-wait.
        """
        while True:
            now = time.monotonic()
            remaining = target_time - now
            if remaining <= 0.0:
                return

            if busy_wait and remaining < 0.002:
                # Busy wait for the last ~2ms if requested.
                continue

            # Sleep a little less than the remaining time to reduce overshoot.
            time.sleep(max(0.0, remaining - 0.001))


def make_threaded_controller_provider(
    controller: BaseController,
    *,
    queue_maxsize: int = 1,
    poll_sleep_s: float = 0.0,
) -> tuple[QueueActionProvider, ControllerWorkerThread]:
    """Convenience helper to build a threaded controller action provider.

    Parameters
    ----------
    controller : BaseController
        Controller to run in the background.
    queue_maxsize : int
        Maximum queue size (usually 1 to keep only the latest action).
    poll_sleep_s : float
        Sleep when no state has been provided yet.

    Returns
    -------
    QueueActionProvider
        Provider to pass into :class:`RealtimeRunner`.
    ControllerWorkerThread
        Worker that must be started and stopped by the caller.
    """
    q: "queue.Queue[Action]" = queue.Queue(maxsize=max(1, int(queue_maxsize)))
    provider = QueueActionProvider(q)
    worker = ControllerWorkerThread(controller, q, poll_sleep_s=poll_sleep_s)
    return provider, worker
