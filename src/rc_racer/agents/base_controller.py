"""
base_controller.py

Abstract base class for all racing controllers.

AGENTS Layer (Competition Layer)
--------------------------------
This module defines the controller interface contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from rc_racer.core.state import State


class BaseController(ABC):
    """
    Abstract controller interface.

    All competitors must subclass this.

    The controller receives the current vehicle state
    and returns an action tuple:

        (acceleration_command, steering_rate_command)

    Notes
    -----
    - Called synchronously in batch mode.
    - May run in separate thread/process in realtime mode.
    - Must NOT control timestep.
    - Must NOT modify State.
    - Must NOT access Environment internals.
    """

    # ==========================================================
    # Lifecycle
    # ==========================================================

    def reset(self) -> None:
        """
        Reset internal controller state.

        Called at the beginning of each episode.

        Notes
        -----
        - Default implementation does nothing.
        - Override if controller maintains memory.
        """
        return None

    # ==========================================================
    # Core API
    # ==========================================================

    @abstractmethod
    def compute_action(
        self,
        state: State,
    ) -> Tuple[float, float]:
        """
        Compute control action from current state.

        Parameters
        ----------
        state : State
            Immutable vehicle state.

        Returns
        -------
        tuple[float, float]
            (acceleration_command, steering_rate_command)

        Notes
        -----
        - Must be deterministic unless explicitly designed otherwise.
        - Must not modify state.
        - Must be side-effect free w.r.t. simulation.
        """
        raise NotImplementedError

    # ==========================================================
    # Optional Realtime Extension
    # ==========================================================

    def get_latest_action(
        self,
        state: State,
    ) -> Tuple[float, float]:
        """
        Optional realtime-compatible API.

        By default, simply calls `compute_action`.

        Realtime runners may override this method
        in threaded/process-based controllers.

        Parameters
        ----------
        state : State

        Returns
        -------
        tuple[float, float]
        """
        return self.compute_action(state)
