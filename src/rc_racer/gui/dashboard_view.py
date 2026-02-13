"""
dashboard_view.py

Passive dashboard renderer for RC Racer.

Displays (configurable):
    • Score
    • Lap time
    • FPS
    • Progress
    • Velocity
    • Steering angle

Extras
------
- Minimal theme preset factory
- Semi-transparent background support (pygame)
- Metric formatting options
- Headless SVG dashboard export (no pygame required)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import pygame

from rc_racer.core.state import State


ColorRGB = Tuple[int, int, int]
ColorRGBA = Tuple[int, int, int, int]
Color = Union[ColorRGB, ColorRGBA]
ThemeName = Literal["dark", "light", "classic"]


# ================================================================
# FORMATTING
# ================================================================


@dataclass(frozen=True)
class DashboardMetricFormat:
    """
    Formatting and visibility options for dashboard metrics.

    Parameters
    ----------
    show_score : bool
        Whether to render the score line.
    show_lap_time : bool
        Whether to render the lap time line.
    show_fps : bool
        Whether to render the FPS line.
    show_progress : bool
        Whether to render the progress line.
    show_velocity : bool
        Whether to render the velocity line.
    show_steering : bool
        Whether to render the steering angle line.

    label_score : str
        Label text for score.
    label_lap_time : str
        Label text for lap time.
    label_fps : str
        Label text for fps.
    label_progress : str
        Label text for progress.
    label_velocity : str
        Label text for velocity.
    label_steering : str
        Label text for steering angle.

    fmt_score : str
        Python format specifier for score (e.g. ".2f").
    fmt_lap_time : str
        Python format specifier for lap time in seconds.
    fmt_fps : str
        Python format specifier for fps.
    fmt_progress : str
        Python format specifier for progress in meters.
    fmt_velocity : str
        Python format specifier for velocity in m/s.
    fmt_steering : str
        Python format specifier for steering angle in rad.

    unit_lap_time : str
        Unit suffix for lap time.
    unit_progress : str
        Unit suffix for progress.
    unit_velocity : str
        Unit suffix for velocity.
    unit_steering : str
        Unit suffix for steering.
    """

    show_score: bool = True
    show_lap_time: bool = True
    show_fps: bool = True
    show_progress: bool = True
    show_velocity: bool = True
    show_steering: bool = True

    label_score: str = "Score"
    label_lap_time: str = "Lap Time"
    label_fps: str = "FPS"
    label_progress: str = "Progress"
    label_velocity: str = "Velocity"
    label_steering: str = "Steering"

    fmt_score: str = ".2f"
    fmt_lap_time: str = ".2f"
    fmt_fps: str = ".1f"
    fmt_progress: str = ".2f"
    fmt_velocity: str = ".2f"
    fmt_steering: str = ".3f"

    unit_lap_time: str = "s"
    unit_progress: str = "m"
    unit_velocity: str = "m/s"
    unit_steering: str = "rad"


# ================================================================
# CONFIG
# ================================================================


@dataclass(frozen=True)
class DashboardViewConfig:
    """
    Rendering configuration for dashboard view.

    Parameters
    ----------
    font_name : str
        Name of pygame font.
    font_size : int
        Font size (px).
    text_color : Color
        RGB or RGBA for text.
    background_color : Color
        RGB or RGBA. If RGBA, alpha is respected via a separate
        alpha surface (semi-transparent panel).
    padding_px : int
        Padding inside dashboard panel.
    line_spacing_px : int
        Vertical spacing between lines.
    panel_position_px : tuple[int, int]
        Top-left screen position of dashboard.
    panel_width_px : int
        Width of dashboard panel.
    metrics : DashboardMetricFormat
        Formatting and visibility options.
    """

    font_name: str
    font_size: int
    text_color: Color
    background_color: Color
    padding_px: int
    line_spacing_px: int
    panel_position_px: Tuple[int, int]
    panel_width_px: int
    metrics: DashboardMetricFormat


# ================================================================
# THEME PRESETS
# ================================================================


def make_dashboard_theme(
    theme: ThemeName = "dark",
    *,
    panel_position_px: Tuple[int, int] = (12, 12),
    panel_width_px: int = 320,
    font_name: str = "consolas",
    font_size: int = 18,
    metrics: Optional[DashboardMetricFormat] = None,
) -> DashboardViewConfig:
    """
    Create a minimal dashboard theme preset.

    Parameters
    ----------
    theme : {"dark", "light", "classic"}
        Theme preset name.
    panel_position_px : tuple[int, int]
        Panel origin.
    panel_width_px : int
        Panel width in pixels.
    font_name : str
        Pygame font name.
    font_size : int
        Font size.
    metrics : DashboardMetricFormat | None
        Optional metric formatting overrides.

    Returns
    -------
    DashboardViewConfig
        Theme configuration.
    """
    m = metrics if metrics is not None else DashboardMetricFormat()

    if theme == "dark":
        return DashboardViewConfig(
            font_name=font_name,
            font_size=font_size,
            text_color=(235, 235, 235),
            background_color=(0, 0, 0, 160),  # semi-transparent
            padding_px=10,
            line_spacing_px=6,
            panel_position_px=panel_position_px,
            panel_width_px=panel_width_px,
            metrics=m,
        )

    if theme == "light":
        return DashboardViewConfig(
            font_name=font_name,
            font_size=font_size,
            text_color=(20, 20, 20),
            background_color=(255, 255, 255, 200),  # semi-transparent
            padding_px=10,
            line_spacing_px=6,
            panel_position_px=panel_position_px,
            panel_width_px=panel_width_px,
            metrics=m,
        )

    # "classic"
    return DashboardViewConfig(
        font_name=font_name,
        font_size=font_size,
        text_color=(255, 255, 0),
        background_color=(0, 0, 64, 200),  # semi-transparent
        padding_px=10,
        line_spacing_px=6,
        panel_position_px=panel_position_px,
        panel_width_px=panel_width_px,
        metrics=m,
    )


# ================================================================
# ABSTRACT BASE
# ================================================================


class BaseDashboardView(ABC):
    """
    Abstract dashboard renderer interface.

    Notes
    -----
    - Must never modify State.
    - Must not contain simulation logic.
    - Must not depend on Environment or Controller.
    - Purely visual.
    """

    @abstractmethod
    def draw(
        self,
        surface: pygame.Surface,
        *,
        state: State,
        score: float,
        lap_time: float,
        fps: float,
    ) -> None:
        """
        Render dashboard.

        Parameters
        ----------
        surface : pygame.Surface
        state : State
            Current immutable vehicle state.
        score : float
            Current accumulated score.
        lap_time : float
            Current lap time in seconds.
        fps : float
            Current frames per second.
        """
        raise NotImplementedError


# ================================================================
# METRIC LINE BUILDER (REUSED BY PYGAME + SVG)
# ================================================================


class DashboardLineBuilder:
    """
    Deterministic dashboard line builder.

    This helper centralizes metric visibility and formatting so both
    pygame rendering and SVG export share identical text output.
    """

    def __init__(self, metrics: DashboardMetricFormat) -> None:
        """
        Initialize line builder.

        Parameters
        ----------
        metrics : DashboardMetricFormat
            Formatting configuration.
        """
        self._m: DashboardMetricFormat = metrics

    def build(
        self,
        *,
        state: State,
        score: float,
        lap_time: float,
        fps: float,
    ) -> list[str]:
        """
        Build formatted lines.

        Parameters
        ----------
        state : State
        score : float
        lap_time : float
        fps : float

        Returns
        -------
        list[str]
            Lines in display order.
        """
        m = self._m
        lines: list[str] = []

        if m.show_score:
            lines.append(f"{m.label_score}: {format(score, m.fmt_score)}")

        if m.show_lap_time:
            lines.append(
                f"{m.label_lap_time}: {format(lap_time, m.fmt_lap_time)} {m.unit_lap_time}"
            )

        if m.show_fps:
            lines.append(f"{m.label_fps}: {format(fps, m.fmt_fps)}")

        if m.show_progress:
            lines.append(
                f"{m.label_progress}: {format(state.progress_s, m.fmt_progress)} {m.unit_progress}"
            )

        if m.show_velocity:
            lines.append(
                f"{m.label_velocity}: {format(state.velocity, m.fmt_velocity)} {m.unit_velocity}"
            )

        if m.show_steering:
            lines.append(
                f"{m.label_steering}: {format(state.steering_angle, m.fmt_steering)} {m.unit_steering}"
            )

        return lines


# ================================================================
# PYGAME IMPLEMENTATION
# ================================================================


class PygameDashboardView(BaseDashboardView):
    """
    Pygame implementation of dashboard view.

    Supports semi-transparent panels by drawing the panel background
    onto an alpha-enabled intermediate surface when needed.
    """

    def __init__(
        self,
        config: DashboardViewConfig,
    ) -> None:
        """
        Initialize dashboard renderer.

        Parameters
        ----------
        config : DashboardViewConfig
            Immutable rendering configuration.
        """
        self._config: DashboardViewConfig = config
        self._font: pygame.font.Font = pygame.font.SysFont(
            config.font_name,
            config.font_size,
        )
        self._line_builder: DashboardLineBuilder = DashboardLineBuilder(config.metrics)

    def draw(
        self,
        surface: pygame.Surface,
        *,
        state: State,
        score: float,
        lap_time: float,
        fps: float,
    ) -> None:
        """
        Draw dashboard panel and metrics.

        Parameters
        ----------
        surface : pygame.Surface
        state : State
        score : float
        lap_time : float
        fps : float
        """
        x0, y0 = self._config.panel_position_px
        lines = self._line_builder.build(
            state=state,
            score=score,
            lap_time=lap_time,
            fps=fps,
        )

        panel_height_px = self._panel_height_px(num_lines=len(lines))
        panel_rect = pygame.Rect(
            x0,
            y0,
            self._config.panel_width_px,
            panel_height_px,
        )

        self._draw_panel_background(surface=surface, rect=panel_rect)

        y_cursor = y0 + self._config.padding_px
        for line in lines:
            text_surface = self._font.render(
                line,
                True,
                self._rgb(self._config.text_color),
            )
            surface.blit(
                text_surface,
                (x0 + self._config.padding_px, y_cursor),
            )
            y_cursor += self._line_advance_px()

    # ------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------

    def _panel_height_px(self, *, num_lines: int) -> int:
        """
        Compute panel height in pixels.

        Parameters
        ----------
        num_lines : int

        Returns
        -------
        int
        """
        return (
            self._config.padding_px * 2
            + num_lines * (self._config.font_size + self._config.line_spacing_px)
        )

    def _line_advance_px(self) -> int:
        """
        Vertical advance per line.

        Returns
        -------
        int
        """
        return self._config.font_size + self._config.line_spacing_px

    def _draw_panel_background(self, *, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """
        Draw background panel, supporting RGBA semi-transparency.

        Parameters
        ----------
        surface : pygame.Surface
        rect : pygame.Rect
        """
        bg = self._config.background_color
        if len(bg) == 3:
            pygame.draw.rect(surface, bg, rect)
            return

        # RGBA: draw on SRCALPHA surface then blit
        r, g, b, a = bg
        panel = pygame.Surface((rect.width, rect.height), flags=pygame.SRCALPHA)
        panel.fill((r, g, b, a))
        surface.blit(panel, rect.topleft)

    @staticmethod
    def _rgb(color: Color) -> ColorRGB:
        """
        Normalize Color to RGB.

        Parameters
        ----------
        color : Color

        Returns
        -------
        tuple[int, int, int]
        """
        if len(color) == 3:
            return color  # type: ignore[return-value]
        r, g, b, _a = color  # type: ignore[misc]
        return (r, g, b)


# ================================================================
# HEADLESS SVG EXPORT
# ================================================================


@dataclass(frozen=True)
class SvgDashboardConfig:
    """
    SVG export configuration.

    Parameters
    ----------
    width_px : int
        SVG viewport width.
    font_family : str
        SVG font-family.
    font_size_px : int
        Text font size.
    text_color : Color
        Text color (RGB or RGBA).
    background_color : Color
        Panel color (RGB or RGBA).
    padding_px : int
        Panel padding.
    line_spacing_px : int
        Additional vertical spacing between lines.
    panel_position_px : tuple[int, int]
        Panel top-left position within the viewport.
    panel_width_px : int
        Panel width.
    metrics : DashboardMetricFormat
        Formatting and visibility options.
    """

    width_px: int
    font_family: str
    font_size_px: int
    text_color: Color
    background_color: Color
    padding_px: int
    line_spacing_px: int
    panel_position_px: Tuple[int, int]
    panel_width_px: int
    metrics: DashboardMetricFormat


class SvgDashboardExporter:
    """
    Headless SVG dashboard exporter.

    Notes
    -----
    - No pygame dependency beyond import presence in this file.
    - Pure string generation; deterministic.
    - Uses same formatting as pygame view.
    """

    def __init__(self, config: SvgDashboardConfig) -> None:
        """
        Initialize SVG exporter.

        Parameters
        ----------
        config : SvgDashboardConfig
        """
        self._config: SvgDashboardConfig = config
        self._line_builder: DashboardLineBuilder = DashboardLineBuilder(config.metrics)

    def to_svg(
        self,
        *,
        state: State,
        score: float,
        lap_time: float,
        fps: float,
    ) -> str:
        """
        Export dashboard as an SVG string.

        Parameters
        ----------
        state : State
        score : float
        lap_time : float
        fps : float

        Returns
        -------
        str
            SVG markup.
        """
        cfg = self._config
        x0, y0 = cfg.panel_position_px

        lines = self._line_builder.build(
            state=state,
            score=score,
            lap_time=lap_time,
            fps=fps,
        )

        line_advance = cfg.font_size_px + cfg.line_spacing_px
        panel_height = cfg.padding_px * 2 + len(lines) * line_advance

        svg_height = max(y0 + panel_height + 1, 1)

        bg_fill, bg_opacity = self._svg_color_and_opacity(cfg.background_color)
        text_fill, text_opacity = self._svg_color_and_opacity(cfg.text_color)

        # Background rect
        rect_svg = (
            f'<rect x="{x0}" y="{y0}" width="{cfg.panel_width_px}" '
            f'height="{panel_height}" rx="6" ry="6" '
            f'fill="{bg_fill}" fill-opacity="{bg_opacity:.4f}" />'
        )

        # Text lines
        text_elements: list[str] = []
        y_cursor = y0 + cfg.padding_px + cfg.font_size_px  # baseline-ish
        for line in lines:
            safe = self._escape_xml(line)
            text_elements.append(
                f'<text x="{x0 + cfg.padding_px}" y="{y_cursor}" '
                f'font-family="{self._escape_xml(cfg.font_family)}" '
                f'font-size="{cfg.font_size_px}" '
                f'fill="{text_fill}" fill-opacity="{text_opacity:.4f}">'
                f"{safe}</text>"
            )
            y_cursor += line_advance

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{cfg.width_px}" height="{svg_height}" '
            f'viewBox="0 0 {cfg.width_px} {svg_height}">'
            f"{rect_svg}"
            f"{''.join(text_elements)}"
            f"</svg>"
        )
        return svg

    @staticmethod
    def _svg_color_and_opacity(color: Color) -> tuple[str, float]:
        """
        Convert RGB/RGBA to SVG fill and opacity.

        Parameters
        ----------
        color : Color

        Returns
        -------
        tuple[str, float]
            (css_color, opacity in [0, 1])
        """
        if len(color) == 3:
            r, g, b = color  # type: ignore[misc]
            return (f"rgb({r},{g},{b})", 1.0)

        r, g, b, a = color  # type: ignore[misc]
        opacity = max(0.0, min(1.0, float(a) / 255.0))
        return (f"rgb({r},{g},{b})", opacity)

    @staticmethod
    def _escape_xml(text: str) -> str:
        """
        Escape basic XML entities.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
