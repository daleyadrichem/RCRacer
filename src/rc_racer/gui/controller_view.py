"""
controller_view.py

GUI layer component for rendering controller-planned trajectory.

GUI Layer
---------
This view is purely visual and renders the predicted trajectory
from the controller onto the screen.

Architecture Compliance
-----------------------
- Does NOT modify simulation state.
- Does NOT access environment internals.
- Only consumes controller output.
- Fully passive renderer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame
from numpy.typing import NDArray

from rc_racer.core.track import Track


FloatArray = NDArray[np.float64]


class ControllerView:
    """
    Renders the planned trajectory from the controller.

    Parameters
    ----------
    track : Track
        Track object used for coordinate scaling.
    pixels_per_meter : float
        Rendering scale.
    color : tuple[int, int, int]
        RGB color for predicted trajectory.
    line_width : int
        Width of trajectory polyline.
    """

    def __init__(
        self,
        *,
        track: Track,
        pixels_per_meter: float,
        color: tuple[int, int, int] = (0, 200, 255),
        line_width: int = 2,
    ) -> None:
        self._track = track
        self._ppm = float(pixels_per_meter)
        self._color = color
        self._line_width = int(line_width)

    # ==========================================================
    # Public API
    # ==========================================================

    def render(
        self,
        *,
        surface: pygame.Surface,
        predicted_path: FloatArray | None,
        track_view,
    ) -> None:
        """
        Render predicted trajectory.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface.
        predicted_path : ndarray or None
            Shape (N, 2) world coordinates.
        track_view : TrackView
            Used for world→screen transformation.
        """
        if predicted_path is None or predicted_path.shape[0] < 2:
            return

        screen_pts = track_view.world_to_screen_points(predicted_path)

        pygame.draw.lines(
            surface,
            self._color,
            False,
            screen_pts,
            self._line_width,
        )
