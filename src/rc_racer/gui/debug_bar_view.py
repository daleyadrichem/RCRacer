"""
debug_bar_view.py

Passive debug bar overlay panel.

GUI Layer
---------
- No simulation logic
- No window creation
- No controller modification
- Renders into existing pygame surface
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import pygame


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class DebugBarConfig:
    """
    Debug bar rendering configuration.

    Parameters
    ----------
    panel_position : tuple[int, int]
        Top-left position of panel.
    panel_size : tuple[int, int]
        Width and height of panel.
    max_abs_value : float
        Maximum absolute value for normalization.
    background_color : Color
        Panel background color.
    text_color : Color
        Text rendering color.
    font_size : int
        Font size.
    """

    panel_position: Tuple[int, int] = (800, 50)
    panel_size: Tuple[int, int] = (360, 280)
    max_abs_value: float = 10.0
    background_color: Color = (20, 20, 20)
    text_color: Color = (240, 240, 240)
    font_size: int = 14


class DebugBarView:
    """
    Debug bar overlay panel with signed values.
    """

    def __init__(self, config: DebugBarConfig | None = None) -> None:
        self._config: DebugBarConfig = config or DebugBarConfig()
        self._font: pygame.font.Font = pygame.font.SysFont(
            "consolas",
            self._config.font_size,
        )

    # ------------------------------------------------------------

    def draw(
        self,
        surface: pygame.Surface,
        acceleration: float,
        u_lat: float,
        u_head: float,
        speed_error: float,
        heading_error: float,
        lateral_error: float,
    ) -> None:
        """
        Draw debug panel with signed bars.

        Parameters
        ----------
        surface : pygame.Surface
        acceleration : float
        u_lat : float
        u_head : float
        speed_error : float
        heading_error : float
        lateral_error : float
        """

        x0, y0 = self._config.panel_position
        width, height = self._config.panel_size

        panel_rect = pygame.Rect(x0, y0, width, height)
        pygame.draw.rect(surface, self._config.background_color, panel_rect)

        values: List[float] = [
            acceleration,
            u_lat,
            u_head,
            speed_error,
            heading_error,
            lateral_error,
        ]

        labels: List[str] = [
            "ACC",
            "U_LAT",
            "U_HEAD",
            "SPD_ERR",
            "HEAD_ERR",
            "LAT_ERR",
        ]

        colors: List[Color] = [
            (0, 200, 255),
            (255, 100, 100),
            (100, 255, 100),
            (200, 200, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        center_y = y0 + height // 2
        usable_half_height = height // 2 - 40
        bar_width = width // (len(values) * 2)

        # zero reference line
        pygame.draw.line(
            surface,
            (120, 120, 120),
            (x0, center_y),
            (x0 + width, center_y),
            1,
        )

        for i, (value, label, color) in enumerate(
            zip(values, labels, colors)
        ):
            normalized = value / self._config.max_abs_value
            normalized = max(min(normalized, 1.0), -1.0)

            bar_height = int(normalized * usable_half_height)

            x = x0 + (i * 2 + 1) * bar_width

            if bar_height >= 0:
                rect = pygame.Rect(
                    x,
                    center_y - bar_height,
                    bar_width,
                    bar_height,
                )
            else:
                rect = pygame.Rect(
                    x,
                    center_y,
                    bar_width,
                    -bar_height,
                )

            pygame.draw.rect(surface, color, rect)

            # numeric label
            text_surface = self._font.render(
                f"{label}: {value:+.3f}",
                True,
                self._config.text_color,
            )

            text_rect = text_surface.get_rect(
                center=(x + bar_width // 2, y0 + 18)
            )

            surface.blit(text_surface, text_rect)
