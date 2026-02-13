"""
app.py

Standalone GUI demo application.

GUI Layer
---------
- Passive visualization only
- No environment stepping
- No controller logic
- No randomness
- Deterministic rendering

This file builds a curved S-track exactly like the
demo_realtime_runner_pid_curved.py example.

Architecture Reference
----------------------
See project architecture specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygame

from rc_racer.core.track import Track
from rc_racer.core.state import State
from rc_racer.gui.track_view import TrackView, TrackViewConfig
from rc_racer.gui.agent_view import PygameAgentView, AgentViewConfig
from rc_racer.gui.dashboard_view import (
    PygameDashboardView,
    make_dashboard_theme,
)

Color = Tuple[int, int, int]


# ================================================================
# TRACK BUILDER
# ================================================================


def build_curved_s_track() -> Track:
    """
    Build a smooth S-shaped track.

    Returns
    -------
    Track
        Immutable track instance.
    """
    xs = np.linspace(0.0, 120.0, 600)

    ys = (
        8.0 * np.sin(0.08 * xs)
        + 4.0 * np.sin(0.18 * xs)
    )

    centerline = np.column_stack((xs, ys)).astype(np.float64)

    return Track(
        centerline=centerline,
        width=10.0,
    )


# ================================================================
# CONFIG
# ================================================================


@dataclass(frozen=True)
class AppConfig:
    """
    GUI configuration container.

    Parameters
    ----------
    width : int
        Window width.
    height : int
        Window height.
    pixels_per_meter : float
        World-to-screen scale.
    background_color : Color
        RGB background color.
    """

    width: int = 1200
    height: int = 700
    pixels_per_meter: float = 6.0
    background_color: Color = (25, 25, 25)


# ================================================================
# APP
# ================================================================


class App:
    """
    Passive GUI application.

    Notes
    -----
    - Does NOT step environment.
    - Does NOT call controller.
    - Receives state snapshots externally.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Initialize GUI.

        Parameters
        ----------
        config : AppConfig
        """
        pygame.init()

        self._config: AppConfig = config
        self._screen: pygame.Surface = pygame.display.set_mode(
            (config.width, config.height)
        )
        pygame.display.set_caption("RC Racer - Curved Track Demo")

        self._clock: pygame.time.Clock = pygame.time.Clock()

        # --------------------------------------------------------
        # Track
        # --------------------------------------------------------

        self._track: Track = build_curved_s_track()
        offset = (100, 400)

        self._track_view = TrackView(
            self._track,
            TrackViewConfig(
                pixels_per_meter=config.pixels_per_meter,
            ),
            screen_offset_px=offset,
        )

        self._agent_view = PygameAgentView(
            AgentViewConfig(
                pixels_per_meter=config.pixels_per_meter,
            ),
            screen_offset_px=offset,
        )

        # --------------------------------------------------------
        # Dashboard
        # --------------------------------------------------------

        dashboard_config = make_dashboard_theme(
            theme="dark",
            panel_position_px=(20, 20),
            panel_width_px=320,
        )

        self._dashboard = PygameDashboardView(dashboard_config)

        # --------------------------------------------------------

        self._current_state: State | None = None
        self._running: bool = False

        # Dashboard metrics (externally updatable)
        self._score: float = 0.0
        self._lap_time: float = 0.0

    # ------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------

    def update_state(
        self,
        state: State,
        *,
        score: float | None = None,
        lap_time: float | None = None,
    ) -> None:
        """
        Update vehicle state snapshot and optional metrics.

        Parameters
        ----------
        state : State
        score : float | None
        lap_time : float | None
        """
        self._current_state = state

        if score is not None:
            self._score = float(score)

        if lap_time is not None:
            self._lap_time = float(lap_time)

    def run(self) -> None:
        """
        Start GUI loop.
        """
        self._running = True

        while self._running:
            self._handle_events()
            self._render()
            self._clock.tick(60)

        pygame.quit()

    def stop(self) -> None:
        """
        Stop GUI loop.
        """
        self._running = False

    # ------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------

    def _handle_events(self) -> None:
        """
        Handle window events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    def _render(self) -> None:
        """
        Render frame.
        """
        self._screen.fill(self._config.background_color)

        self._track_view.draw(self._screen)

        if self._current_state is not None:
            self._agent_view.draw(self._screen, self._current_state)

            # Dashboard
            self._dashboard.draw(
                self._screen,
                state=self._current_state,
                score=self._score,
                lap_time=self._lap_time,
                fps=self._clock.get_fps(),
            )

        pygame.display.flip()
