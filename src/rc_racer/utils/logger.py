"""
logger.py

Deterministic replay logger for the racing simulation.

This module provides ReplayLogger, which logs:

- state
- action
- reward
- done

The logger:

- Is deterministic
- Contains no randomness
- Has no timestamps
- Does not depend on GUI
- Does not depend on controllers
- Is safe for multiprocessing usage

Log format:
JSON Lines (one JSON object per step)

This module belongs to the UTILS layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, TextIO

from rc_racer.core.state import State, StateArray


class ReplayLogger:
    """
    Deterministic replay logger.

    Parameters
    ----------
    path : str | Path
        Output log file path.
    """

    def __init__(self, path: str | Path) -> None:
        self._path: Path = Path(path)
        self._file: TextIO = self._path.open("w", encoding="utf-8")

    # ------------------------------------------------------------------

    def log_step(
        self,
        *,
        state: State | StateArray,
        action: Dict[str, Any],
        reward: float,
        done: bool,
    ) -> None:
        """
        Log a single simulation step.

        Parameters
        ----------
        state : State | StateArray
            Current simulation state.
        action : dict
            Action applied.
        reward : float
            Reward received.
        done : bool
            Termination flag.
        """
        record: Dict[str, Any] = {
            "state": state.to_dict(),
            "action": action,
            "reward": reward,
            "done": done,
        }

        self._file.write(json.dumps(record))
        self._file.write("\n")

    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Close log file.
        """
        self._file.close()

    # ------------------------------------------------------------------

    @staticmethod
    def replay(path: str | Path) -> Iterable[Dict[str, Any]]:
        """
        Replay log file.

        Parameters
        ----------
        path : str | Path

        Returns
        -------
        Iterable[dict]
            Sequence of logged step dictionaries.
        """
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
