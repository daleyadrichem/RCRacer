"""
Passive pygame-compatible renderer for an "ideal" racing line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pygame
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class RacingLineViewConfig:
    """
    Rendering config for RacingLineView.

    Parameters
    ----------
    color : Color
        RGB color of the line.
    width_px : int
        Line thickness in pixels.
    pixels_per_meter : float
        World->screen scale.
    """

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
        """
        Initialize the racing line view.

        Parameters
        ----------
        path_points : ndarray of shape (M, 2)
            Ideal path polyline in world meters.
        config : RacingLineViewConfig | None
            Rendering config.
        screen_offset_px : tuple[int, int]
            Pixel offset applied after scaling.
        """
        self._path: FloatArray = np.asarray(path_points, dtype=np.float64)
        self._cfg: RacingLineViewConfig = config if config is not None else RacingLineViewConfig()
        self._offset_px: Tuple[int, int] = screen_offset_px

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
        """
        Draw the ideal line on a pygame surface.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface.
        """
        if self._path.shape[0] < 2:
            return
        pts = self._world_to_screen_points(self._path)
        pygame.draw.lines(surface, self._cfg.color, False, pts, int(self._cfg.width_px))
