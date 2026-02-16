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

from rc_racer.core.track_factory import TrackFactory
from rc_racer.gui.track_view import TrackView, TrackViewConfig


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

def main() -> None:
    """
    Entry point for track demo.
    """
    track = TrackFactory.create("f1_like_closed")
    config = WindowConfig()

    # ------------------------------------------------------------
    # Compute world bounding box
    # ------------------------------------------------------------
    all_points = np.vstack(
        (
            track.centerline,
            track.left_boundary,
            track.right_boundary,
        )
    )

    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)

    world_width = float(max_xy[0] - min_xy[0])
    world_height = float(max_xy[1] - min_xy[1])

    # ------------------------------------------------------------
    # Compute zoom
    # ------------------------------------------------------------
    margin_factor = 0.9

    ppm_x = config.width * margin_factor / world_width
    ppm_y = config.height * margin_factor / world_height

    pixels_per_meter = min(ppm_x, ppm_y)

    # ------------------------------------------------------------
    # Center track
    # ------------------------------------------------------------
    world_center = 0.5 * (min_xy + max_xy)

    screen_center_x = config.width // 2
    screen_center_y = config.height // 2

    offset_x = int(screen_center_x - world_center[0] * pixels_per_meter)
    offset_y = int(screen_center_y + world_center[1] * pixels_per_meter)

    offset = (offset_x, offset_y)
   
    # ------------------------------------------------------------
    # Create view
    # ------------------------------------------------------------
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("Simple Track Demo")

    clock = pygame.time.Clock()

    view = TrackView(
        track=track,
        config=TrackViewConfig(
            pixels_per_meter=pixels_per_meter,
        ),
        screen_offset_px=offset,
    )

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