"""
show_simple_track.py

Simple runnable demo that visualizes a deterministic circular track.

This module:
- Belongs outside the architecture layers (example/demo)
- Does not contain simulation logic
- Does not modify Track
- Does not introduce randomness

Run with:
    python -m examples.show_simple_track
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import pygame

from rc_racer.core.track import Track
from rc_racer.gui.track_view import TrackView


FloatArray = NDArray[np.float64]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class WindowConfig:
    """
    Window configuration.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    background_color : Color
        Background RGB color.
    fps : int
        Target frames per second.
    """
    width: int = 1200
    height: int = 1200
    background_color: Color = (25, 25, 25)
    fps: int = 60


def create_circular_centerline(
    center: Tuple[float, float],
    radius: float,
    num_points: int,
) -> FloatArray:
    """
    Create circular centerline.

    Parameters
    ----------
    center : Tuple[float, float]
        Center of circle.
    radius : float
        Circle radius.
    num_points : int
        Number of discretization points.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (N, 2).
    """
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)

    return np.column_stack((x, y)).astype(np.float64)

def create_challenging_centerline(
    center: Tuple[float, float],
    base_radius: float,
    num_points: int,
) -> FloatArray:
    """
    Create a deterministic challenging closed track centerline.

    This track is a sinusoidally modulated circular loop that
    introduces varying curvature and technical sections while
    remaining smooth and non-self-intersecting.

    Parameters
    ----------
    center : Tuple[float, float]
        Center of track.
    base_radius : float
        Base radius of track.
    num_points : int
        Number of discretization points.

    Returns
    -------
    NDArray[np.float64]
        Centerline array of shape (N, 2).
    """
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    # --- Deterministic curvature modulation ---
    radius = (
        base_radius
        + 60.0 * np.sin(2.0 * angles)        # long sweeping bends
        + 35.0 * np.sin(5.0 * angles + 0.5)  # medium technical sections
        + 20.0 * np.sin(9.0 * angles)        # tighter kinks
    )

    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)

    return np.column_stack((x, y)).astype(np.float64)

def main() -> None:
    """
    Entry point for track demo.
    """
    config = WindowConfig()

    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("Simple Track Demo")

    clock = pygame.time.Clock()

    centerline = create_challenging_centerline(
        center=(config.width / 2.0, config.height / 2.0),
        base_radius=400.0,
        num_points=800,
    )


    track = Track(
        centerline=centerline,
        width=80.0,
    )

    view = TrackView(track)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(config.background_color)
        view.draw(screen)
        pygame.display.flip()
        clock.tick(config.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
