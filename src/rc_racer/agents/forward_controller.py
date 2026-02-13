"""
forward_controller.py

Simple controller that drives straight forward.

AGENTS Layer
------------
- No timestep control
- No environment access
- No randomness
- Fully deterministic
"""

from __future__ import annotations

from typing import Tuple

from rc_racer.core.state import State
from rc_racer.agents.base_controller import BaseController


class ForwardController(BaseController):
    """
    Drives forward with constant acceleration.
    """

    def __init__(
        self,
        acceleration: float = 1.0,
        steering_rate: float = 0.0,
    ) -> None:
        self._acceleration = acceleration
        self._steering_rate = steering_rate

    def compute_action(
        self,
        state: State,
    ) -> Tuple[float, float]:
        return self._acceleration, self._steering_rate
