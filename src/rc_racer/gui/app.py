"""
rc_racer.gui.app

Standalone passive GUI application for RC Racer.

GUI Layer
---------
- Passive visualization only
- No environment stepping
- No controller logic
- No randomness
- Deterministic rendering

Architecture
------------
- Simulation loop is authoritative.
- GUI only receives immutable state snapshots.
- Track is injected (no construction inside GUI).
- Never modifies simulation state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import pygame

from rc_racer.core.track import Track
from rc_racer.core.state import State
from rc_racer.gui.track_view import TrackView, TrackViewConfig
from rc_racer.gui.agent_view import PygameAgentView, AgentViewConfig
from rc_racer.gui.dashboard_view import (
    PygameDashboardView,
    make_dashboard_theme,
)
from rc_racer.gui.debug_bar_view import DebugBarView, DebugBarConfig


Color = Tuple[int, int, int]


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class AppConfig:
    """
    GUI configuration container.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    pixels_per_meter : float
        World-to-screen scaling factor.
    background_color : tuple[int, int, int]
        RGB background color.
    screen_offset_px : tuple[int, int]
        Pixel offset for world origin placement.
    window_title : str
        Window caption.
    show_debug : bool
        Whether to render debug bar overlay.
    """

    width: int = 1200
    height: int = 700
    pixels_per_meter: float = 6.0
    background_color: Color = (25, 25, 25)
    screen_offset_px: Tuple[int, int] = (100, 400)
    window_title: str = "RC Racer"
    show_debug: bool = False


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
    - Safe to run alongside RealtimeRunner.
    """

    # ------------------------------------------------------------

    def __init__(
        self,
        track: Track,
        config: AppConfig,
    ) -> None:
        pygame.init()

        self._track: Track = track
        self._config: AppConfig = config

        self._screen: pygame.Surface = pygame.display.set_mode(
            (config.width, config.height)
        )
        pygame.display.set_caption(config.window_title)

        self._clock: pygame.time.Clock = pygame.time.Clock()

        # Track View
        self._track_view = TrackView(
            track=self._track,
            config=TrackViewConfig(
                pixels_per_meter=config.pixels_per_meter,
            ),
            screen_offset_px=config.screen_offset_px,
        )

        # Agent View
        self._agent_view = PygameAgentView(
            AgentViewConfig(
                pixels_per_meter=config.pixels_per_meter,
            ),
            screen_offset_px=config.screen_offset_px,
        )

        # Dashboard
        dashboard_config = make_dashboard_theme(
            theme="dark",
            panel_position_px=(20, 20),
            panel_width_px=320,
        )
        self._dashboard = PygameDashboardView(dashboard_config)

        # Debug Bar (optional)
        self._debug_view: Optional[DebugBarView] = None
        panel_width = 300
        panel_height = 240
        margin = 20

        self._debug_view = DebugBarView(
            DebugBarConfig(
                panel_position=(
                    config.width - panel_width - margin,
                    margin,
                ),
                panel_size=(panel_width, panel_height),
                max_abs_value=10.0,
            )
        )

        # State
        self._current_state: Optional[State] = None
        self._score: float = 0.0
        self._lap_time: float = 0.0

        # Debug values
        self._debug_accel: float = 0.0
        self._debug_u_lat: float = 0.0
        self._debug_u_head: float = 0.0
        self._debug_speed_error: float = 0.0
        self._debug_lateral_error: float = 0.0
        self._debug_heading_error: float = 0.0

        self._running: bool = False

    # ============================================================
    # PUBLIC API
    # ============================================================

    def update_state(
        self,
        state: State,
        *,
        score: float | None = None,
        lap_time: float | None = None,
        debug_values: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        """
        Update GUI snapshot.

        Parameters
        ----------
        state : State
            Immutable state from simulation.
        score : float | None
            Optional score update.
        lap_time : float | None
            Optional lap time update.
        debug_values : tuple[float, float, float] | None
            Optional (acceleration, u_lat, u_head, speed_error, heading_error, lateral_error ).
        """
        self._current_state = state

        if score is not None:
            self._score = float(score)

        if lap_time is not None:
            self._lap_time = float(lap_time)

        if debug_values is not None:
            self._debug_accel = float(debug_values[0])
            self._debug_u_lat = float(debug_values[1])
            self._debug_u_head = float(debug_values[2])
            self._debug_speed_error = float(debug_values[3])
            self._debug_lateral_error = float(debug_values[4])
            self._debug_heading_error = float(debug_values[5])

    # ------------------------------------------------------------

    def run(self) -> None:
        self._running = True

        while self._running:
            self._handle_events()
            self._render()
            self._clock.tick(60)

        pygame.quit()

    # ------------------------------------------------------------

    def stop(self) -> None:
        self._running = False

    # ============================================================
    # INTERNALS
    # ============================================================

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

    # ------------------------------------------------------------

    def _render(self) -> None:
        self._screen.fill(self._config.background_color)

        self._track_view.draw(self._screen)

        if self._current_state is not None:
            self._agent_view.draw(self._screen, self._current_state)

            self._dashboard.draw(
                self._screen,
                state=self._current_state,
                score=self._score,
                lap_time=self._lap_time,
                fps=self._clock.get_fps(),
            )

            if self._debug_view is not None:
                self._debug_view.draw(
                    self._screen,
                    acceleration=self._debug_accel,
                    u_lat=self._debug_u_lat,
                    u_head=self._debug_u_head,
                    speed_error=self._debug_speed_error,
                    heading_error=self._debug_heading_error,
                    lateral_error=self._debug_lateral_error
                )

        pygame.display.flip()
