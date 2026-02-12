"""
track_view.py

Passive visualization component for rendering a Track instance.

This module belongs to the GUI layer and must:
- Never modify simulation state
- Never contain simulation logic
- Never introduce randomness
- Only read immutable Track data

It renders:
- Left boundary
- Right boundary
- Centerline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame
import numpy as np
from numpy.typing import NDArray

from rc_racer.core.track import Track


FloatArray = NDArray[np.float64]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class TrackViewConfig:
    """
    Rendering configuration for TrackView.

    Parameters
    ----------
    centerline_color : Color
        RGB color for centerline.
    boundary_color : Color
        RGB color for boundaries.
    centerline_width : int
        Line width for centerline.
    boundary_width : int
        Line width for boundaries.
    """
    centerline_color: Color = (180, 180, 180)
    boundary_color: Color = (255, 255, 255)
    centerline_width: int = 2
    boundary_width: int = 3


class TrackView:
    """
    Passive renderer for Track.

    Notes
    -----
    - Does not modify Track.
    - Does not depend on simulation timing.
    - Safe to recreate per frame.
    """

    def __init__(
        self,
        track: Track,
        config: TrackViewConfig | None = None,
    ) -> None:
        """
        Initialize TrackView.

        Parameters
        ----------
        track : Track
            Immutable core Track object.
        config : TrackViewConfig | None
            Optional rendering configuration.
        """
        self._track: Track = track
        self._config: TrackViewConfig = config or TrackViewConfig()

        # Cached references (read-only)
        self._centerline: FloatArray = self._track.centerline
        self._left: FloatArray = self._track.left_boundary
        self._right: FloatArray = self._track.right_boundary

    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw track on a pygame surface.

        Parameters
        ----------
        surface : pygame.Surface
            Rendering surface.
        """
        self._draw_polyline(
            surface,
            self._left,
            self._config.boundary_color,
            self._config.boundary_width,
        )

        self._draw_polyline(
            surface,
            self._right,
            self._config.boundary_color,
            self._config.boundary_width,
        )

        self._draw_polyline(
            surface,
            self._centerline,
            self._config.centerline_color,
            self._config.centerline_width,
        )

    @staticmethod
    def _draw_polyline(
        surface: pygame.Surface,
        points: FloatArray,
        color: Color,
        width: int,
    ) -> None:
        """
        Draw polyline from Nx2 array.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface.
        points : NDArray[np.float64]
            Polyline points of shape (N, 2).
        color : Color
            RGB color.
        width : int
            Line thickness.
        """
        if points.shape[0] < 2:
            return

        pygame.draw.lines(
            surface,
            color,
            False,
            points.tolist(),
            width,
        )
