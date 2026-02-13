"""
track_view.py

Passive visualization component for rendering a Track instance.

GUI Layer
---------
This module:
- Never modifies simulation state
- Never contains simulation logic
- Never introduces randomness
- Only reads immutable Track data

It renders:
- Left boundary
- Right boundary
- Centerline

World → Screen transformation is handled internally using:
    screen_x = (x * pixels_per_meter) + offset_x
    screen_y = (-y * pixels_per_meter) + offset_y

This ensures consistency with AgentView.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import pygame
import numpy as np
from numpy.typing import NDArray

from rc_racer.core.track import Track


FloatArray = NDArray[np.float64]
Color = Tuple[int, int, int]


# ================================================================
# CONFIGURATION
# ================================================================


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
    pixels_per_meter : float
        Scale factor from world meters to screen pixels.
    """

    centerline_color: Color = (180, 180, 180)
    boundary_color: Color = (255, 255, 255)
    centerline_width: int = 2
    boundary_width: int = 3
    pixels_per_meter: float = 12.0


# ================================================================
# TRACK VIEW
# ================================================================


class TrackView:
    """
    Passive renderer for Track.

    Notes
    -----
    - Does not modify Track.
    - Does not depend on simulation timing.
    - Safe to recreate per frame.
    - Performs world→screen coordinate transform internally.
    """

    # ------------------------------------------------------------

    def __init__(
        self,
        track: Track,
        config: TrackViewConfig | None = None,
        screen_offset_px: Tuple[int, int] = (0, 0),
    ) -> None:
        """
        Initialize TrackView.

        Parameters
        ----------
        track : Track
            Immutable core Track object.
        config : TrackViewConfig | None
            Optional rendering configuration.
        screen_offset_px : tuple[int, int]
            Pixel offset applied after scaling (useful to place origin).
        """
        self._track: Track = track
        self._config: TrackViewConfig = config or TrackViewConfig()
        self._offset_px: Tuple[int, int] = screen_offset_px

        # Cached immutable references
        self._centerline: FloatArray = self._track.centerline
        self._left: FloatArray = self._track.left_boundary
        self._right: FloatArray = self._track.right_boundary

    # ------------------------------------------------------------
    # World → Screen transform
    # ------------------------------------------------------------

    def _world_to_screen_points(
        self,
        points_world: FloatArray,
    ) -> List[Tuple[int, int]]:
        """
        Convert world coordinates to pygame screen integer coordinates.

        Parameters
        ----------
        points_world : ndarray of shape (N, 2)

        Returns
        -------
        list[tuple[int, int]]
            Screen coordinates.
        """
        ppm: float = self._config.pixels_per_meter
        ox, oy = self._offset_px

        pts: List[Tuple[int, int]] = []
        for x, y in points_world:
            sx: int = int(x * ppm + ox)
            sy: int = int(-y * ppm + oy)
            pts.append((sx, sy))

        return pts

    # ------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------

    def _draw_polyline(
        self,
        surface: pygame.Surface,
        points_world: FloatArray,
        color: Color,
        width: int,
    ) -> None:
        """
        Draw polyline from Nx2 world-coordinate array.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface.
        points_world : ndarray of shape (N, 2)
            World coordinates.
        color : Color
            RGB color.
        width : int
            Line thickness.
        """
        if points_world.shape[0] < 2:
            return

        screen_pts = self._world_to_screen_points(points_world)

        pygame.draw.lines(
            surface,
            color,
            False,
            screen_pts,
            width,
        )
