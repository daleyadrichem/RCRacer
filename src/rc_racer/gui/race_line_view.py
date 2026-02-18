"""
Passive pygame-compatible renderer for an optimized racing line.
Supports new lap_time_optimizer JSON format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import json

import pygame
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class RacingLineViewConfig:
    color: Color = (255, 80, 80)
    width_px: int = 4
    pixels_per_meter: float = 12.0


class PygameRacingLineView:
    """
    Passive renderer for an optimized racing line polyline.
    """

    def __init__(
        self,
        *,
        path_points: FloatArray,
        config: RacingLineViewConfig | None = None,
        screen_offset_px: Tuple[int, int] = (0, 0),
    ) -> None:
        self._path: FloatArray = np.asarray(path_points, dtype=np.float64)
        self._cfg: RacingLineViewConfig = config if config is not None else RacingLineViewConfig()
        self._offset_px: Tuple[int, int] = screen_offset_px

    # ------------------------------------------------------------
    # NEW: Load from lap_time_optimizer JSON
    # ------------------------------------------------------------
    @classmethod
    def from_json(
        cls,
        json_path: str | Path,
        *,
        config: RacingLineViewConfig | None = None,
        screen_offset_px: Tuple[int, int] = (0, 0),
    ) -> "PygameRacingLineView":

        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(p)

        with p.open("r") as f:
            data = json.load(f)

        traj = data.get("trajectory", [])
        if len(traj) == 0:
            raise ValueError("JSON contains no trajectory")

        pts = np.array([[pt["x"], pt["y"]] for pt in traj], dtype=np.float64)

        return cls(
            path_points=pts,
            config=config,
            screen_offset_px=screen_offset_px,
        )

    # ------------------------------------------------------------

    def _world_to_screen_points(self, pts: FloatArray) -> List[Tuple[int, int]]:
        ppm = float(self._cfg.pixels_per_meter)
        ox, oy = self._offset_px

        out: List[Tuple[int, int]] = []
        for x, y in pts:
            sx = int(float(x) * ppm + ox)
            sy = int(-float(y) * ppm + oy)
            out.append((sx, sy))
        return out

    def draw(self, surface: pygame.Surface) -> None:
        if self._path.shape[0] < 2:
            return

        pts = self._world_to_screen_points(self._path)
        pygame.draw.lines(surface, self._cfg.color, False, pts, int(self._cfg.width_px))
