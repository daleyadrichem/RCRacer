"""
Realtime (wall-clock paced) simulation runner.

This module implements the "Realtime Demo" execution mode:

- The environment is stepped synchronously and deterministically.
- The simulation loop is authoritative.
- Wall-clock sleeping is used only for pacing and never affects determinism.
- Controllers may run synchronously OR asynchronously (thread/process), but must
  never access the environment directly.

Key Contract
------------
If the controller takes longer than one simulation step to compute an action,
the runner continues stepping with the last available action (sample-and-hold).

Notes
-----
This file follows the Simulation-layer architectural rules:
- Simulation must never depend on controller speed or GUI.
- Environment remains single-threaded and synchronous.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Callable, Protocol, Tuple, runtime_checkable

from rc_racer.agents.base_controller import BaseController
from rc_racer.environment.environment import Environment
from rc_racer.core.state import State

Action = Tuple[float, float]

# =============================================================================
# Configuration + Result
# =============================================================================


@dataclass(frozen=True)
class RunnerConfig:
    """Configuration for realtime execution.

    Parameters
    ----------
    target_fps : float | None
        Wall-clock pacing target. If None, do not sleep (run as fast as possible).
        This does not affect determinism.
    max_steps : int | None
        Optional hard cap on environment steps for the run.
    busy_wait : bool
        If True, use a short busy-wait for sub-millisecond timing accuracy.
        This increases CPU usage.
    """

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
    terminated : bool
        Whether the environment terminated the episode.
    """

    steps: int
    total_reward: float
    wall_time_s: float
    achieved_fps: float
    terminated: bool


# =============================================================================
# Providers
# =============================================================================

class ThreadedControllerProvider():
    """Asynchronous provider that computes actions in a background thread.

    Design
    ------
    - The runner remains authoritative and never blocks on action computation.
    - The runner pushes state snapshots to this provider via `push_state`.
    - The worker computes `controller.compute_action(state)` and publishes
      actions into a size-1 queue (keeps only the latest).
    - `get_action` drains the queue non-blockingly and returns the most recent
      action, re-using the last action if the worker is late.

    Parameters
    ----------
    controller : BaseController
        Controller instance used by the background worker.
    default_action : Action
        Initial action applied until the first action is produced.
    poll_timeout_s : float
        Worker wait timeout while waiting for state updates.
        Small non-zero values reduce CPU usage.
    queue_maxsize : int
        Action queue size. Use 1 to keep only the latest action.
    """

    def __init__(
        self,
        controller: BaseController,
        *,
        default_action: Action = (0.0, 0.0),
        poll_timeout_s: float = 0.01,
        queue_maxsize: int = 1,
    ) -> None:
        self._controller: BaseController = controller

        self._latest_action: Action = (float(default_action[0]), float(default_action[1]))
        self._q: "queue.Queue[Action]" = queue.Queue(maxsize=max(1, int(queue_maxsize)))

        self._state_lock = threading.Lock()
        self._latest_state: State | None = None

        self._state_evt = threading.Event()
        self._stop_evt = threading.Event()

        self._poll_timeout_s: float = float(max(0.0, poll_timeout_s))
        self._thread = threading.Thread(target=self._run, daemon=True)

    # --------------------------
    # Lifecycle
    # --------------------------

    def start(self) -> None:
        """Start the worker thread."""
        self._controller.reset()
        self._stop_evt.clear()
        self._thread.start()

    def stop(self) -> None:
        """Stop the worker thread."""
        self._stop_evt.set()
        self._state_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # --------------------------
    # Provider API
    # --------------------------

    def reset(self) -> None:
        """Reset provider state for a new episode."""
        self._latest_action = (0.0, 0.0)
        with self._state_lock:
            self._latest_state = None

        # Drain action queue best-effort
        while True:
            try:
                _ = self._q.get_nowait()
            except Exception:
                break

    def push_state(self, state: State) -> None:
        """Provide the newest state snapshot to the worker."""
        with self._state_lock:
            self._latest_state = state
        self._state_evt.set()

    def get_action(self, state: State) -> Action:
        """Return the latest available action (never blocks)."""
        del state  # state is provided separately via push_state()

        # Drain queue to keep only the most recent action
        while True:
            try:
                a = self._q.get_nowait()
            except Exception:
                break
            self._latest_action = (float(a[0]), float(a[1]))

        return self._latest_action

    # --------------------------
    # Worker
    # --------------------------

    def _run(self) -> None:
        """Worker loop."""
        while not self._stop_evt.is_set():
            # Wait for a state update (or timeout so we can check stop flag)
            self._state_evt.wait(timeout=self._poll_timeout_s)
            self._state_evt.clear()

            if self._stop_evt.is_set():
                return

            with self._state_lock:
                state = self._latest_state

            if state is None:
                continue

            action = self._controller.compute_action(state)
            action_out: Action = (float(action[0]), float(action[1]))

            # Publish latest action best-effort, preserving "latest wins"
            try:
                self._q.put_nowait(action_out)
            except queue.Full:
                try:
                    _ = self._q.get_nowait()
                except Exception:
                    pass
                try:
                    self._q.put_nowait(action_out)
                except Exception:
                    pass


# =============================================================================
# Runner
# =============================================================================


class RealtimeRunner:
    """Authoritative realtime simulation runner.

    Parameters
    ----------
    env : Environment
        Environment instance (single-threaded, synchronous).
    action_provider : ThreadedControllerProvider
        Source of actions. May be asynchronous.
    config : RunnerConfig
        Wall-clock pacing + step configuration.
    on_step : Callable[[State, float, bool, dict], None] | None
        Optional callback invoked after every environment step (e.g. GUI updates).
        This must not affect simulation determinism.
    """

    def __init__(
        self,
        env: Environment,
        action_provider: ThreadedControllerProvider,
        config: RunnerConfig,
        *,
        on_step: Callable[[State, float, bool, dict], None] | None = None,
    ) -> None:
        self._env: Environment = env
        self._provider: ThreadedControllerProvider = action_provider
        self._config: RunnerConfig = config
        self._on_step = on_step

        self._stop_flag: bool = False

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
        loop_start = time.monotonic()

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

            self._provider.push_state(state)  # pyright: ignore[reportGeneralTypeIssues]

            # Get action immediately (must not block). If controller is late,
            # providers must reuse the last action.
            t0 = time.monotonic()
            action = self._provider.get_action(state)
            t1 = time.monotonic()
            print("Solve time:", t1 - t0)

            next_state, reward, done, info = self._env.step(action)
            steps += 1
            total_reward += float(reward)

            if self._on_step is not None:
                self._on_step(next_state, float(reward), bool(done), info)

            # Wall-clock pacing only; does not affect determinism
            if target_fps is not None:
                t_next += target_period_s
                self._sleep_until(t_next, busy_wait=self._config.busy_wait)

        wall_time_s = max(1e-12, time.monotonic() - t_start)
        achieved_fps = float(steps) / wall_time_s

        return EpisodeResult(
            steps=int(steps),
            total_reward=float(total_reward),
            wall_time_s=float(wall_time_s),
            achieved_fps=float(achieved_fps),
            terminated=bool(done),
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

            # Busy wait only at the very end for better precision.
            if busy_wait and remaining < 0.002:
                continue

            # Sleep slightly less than remaining to reduce overshoot.
            time.sleep(max(0.0, remaining - 0.001))
