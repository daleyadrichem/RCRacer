"""
agent_view.py

Passive pygame-compatible vehicle renderer with:
- wheel visualization (4 wheels)
- drift angle visualization (velocity direction + arc)

Architectural Rules
-------------------
- Reads State only
- No simulation logic
- No controller dependency
- No randomness
- No state mutation

Architecture reference:
:contentReference[oaicite:2]{index=2}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygame

from rc_racer.core.state import State


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class AgentViewConfig:
    """
    Rendering configuration.

    Parameters
    ----------
    body_color : Color
        RGB body color.
    heading_color : Color
        RGB heading vector color.
    velocity_color : Color
        RGB velocity vector color.
    drift_arc_color : Color
        RGB drift arc color.
    wheel_color : Color
        RGB wheel color.

    body_length : float
        Vehicle body length (meters).
    body_width : float
        Vehicle body width (meters).

    wheelbase : float
        Axle distance used for wheel placement in rendering (meters).
        This is purely visual and does not affect physics.

    rear_axle_ratio : float
        Ratio lr/L used to estimate kinematic slip angle beta for drift visualization.

    wheel_length : float
        Wheel length (meters).
    wheel_width : float
        Wheel width (meters).

    pixels_per_meter : float
        Scale factor from meters to pixels.

    show_heading : bool
        Whether to draw heading indicator.
    show_velocity : bool
        Whether to draw velocity indicator.
    show_drift : bool
        Whether to draw drift angle arc and velocity-direction line.
    show_wheels : bool
        Whether to draw wheels.
    """

    body_color: Color = (0, 150, 255)
    heading_color: Color = (255, 0, 0)
    velocity_color: Color = (0, 255, 0)
    drift_arc_color: Color = (255, 255, 0)
    wheel_color: Color = (30, 30, 30)

    body_length: float = 4.0
    body_width: float = 2.0

    wheelbase: float = 2.5
    rear_axle_ratio: float = 0.5

    wheel_length: float = 0.6
    wheel_width: float = 0.25

    pixels_per_meter: float = 12.0

    show_heading: bool = True
    show_velocity: bool = True
    show_drift: bool = True
    show_wheels: bool = True


class BaseAgentView(ABC):
    """
    Abstract agent renderer.
    """

    @abstractmethod
    def draw(self, surface: pygame.Surface, state: State) -> None:
        """
        Draw the agent on a pygame surface.

        Parameters
        ----------
        surface : pygame.Surface
            Target rendering surface.
        state : State
            Current vehicle state snapshot.
        """
        raise NotImplementedError


class PygameAgentView(BaseAgentView):
    """
    Pygame implementation of AgentView.

    Notes
    -----
    - Pure visualization: reads State only.
    - Uses a simple world->screen transform:
      screen_x = (x * ppm) + offset_x
      screen_y = (-y * ppm) + offset_y
    - Optional screen offset can be set to position world origin on screen.
    """

    def __init__(self, config: AgentViewConfig | None = None, screen_offset_px: Tuple[int, int] = (0, 0)) -> None:
        """
        Initialize the pygame agent renderer.

        Parameters
        ----------
        config : AgentViewConfig | None
            Optional rendering config.
        screen_offset_px : tuple[int, int]
            Pixel offset applied after scaling (useful to place origin).
        """
        self._config = config or AgentViewConfig()
        self._offset_px = screen_offset_px

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _rot(self, theta: float) -> np.ndarray:
        """
        Rotation matrix for 2D.

        Parameters
        ----------
        theta : float
            Angle in radians.

        Returns
        -------
        ndarray of shape (2, 2)
        """
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    def _world_to_screen_points(self, points_world: np.ndarray) -> list[Tuple[int, int]]:
        """
        Convert world coordinates to pygame screen integer coordinates.

        Parameters
        ----------
        points_world : ndarray of shape (N, 2)

        Returns
        -------
        list[tuple[int, int]]
        """
        ppm = self._config.pixels_per_meter
        ox, oy = self._offset_px

        pts = []
        for x, y in points_world:
            sx = int(x * ppm + ox)
            sy = int(-y * ppm + oy)
            pts.append((sx, sy))
        return pts

    def _vehicle_body_polygon(self, state: State) -> np.ndarray:
        """
        Vehicle body rectangle anchored at rear axle.

        Returns
        -------
        ndarray of shape (4, 2)
        """
        L = self._config.body_length
        W = self._config.body_width

        corners_local = np.array(
            [
                [0.0, -W / 2.0],
                [L, -W / 2.0],
                [L, W / 2.0],
                [0.0, W / 2.0],
            ],
            dtype=np.float64,
        )

        R = self._rot(state.heading)
        return corners_local @ R.T + np.array([state.x, state.y], dtype=np.float64)

    def _wheel_rect(self, center_local: np.ndarray, angle: float) -> np.ndarray:
        """
        Create a wheel rectangle in vehicle-local coordinates, then rotate by `angle`.

        Parameters
        ----------
        center_local : ndarray of shape (2,)
            Wheel center in vehicle-local frame.
        angle : float
            Wheel orientation angle relative to vehicle frame (radians).

        Returns
        -------
        ndarray of shape (4, 2)
            Wheel corners in vehicle-local frame.
        """
        wl = self._config.wheel_length
        ww = self._config.wheel_width

        # rectangle centered at origin then translated to center_local
        rect = np.array(
            [
                [-wl / 2.0, -ww / 2.0],
                [wl / 2.0, -ww / 2.0],
                [wl / 2.0, ww / 2.0],
                [-wl / 2.0, ww / 2.0],
            ],
            dtype=np.float64,
        )

        Rw = self._rot(angle)
        return (rect @ Rw.T) + center_local

    def _wheel_polygons_world(self, state: State) -> list[np.ndarray]:
        """
        Compute four wheel polygons in world frame.

        Returns
        -------
        list[ndarray]
            Each entry is ndarray of shape (4, 2) in world coordinates.
        """
        wb = self._config.wheelbase
        track_half = 0.42 * self._config.body_width  # purely visual
        # rear axle at x=0, front axle at x=wheelbase
        rear_center_x = 0.0
        front_center_x = wb

        # wheel centers in vehicle-local frame
        centers = [
            np.array([rear_center_x, -track_half], dtype=np.float64),  # rear-left
            np.array([rear_center_x, track_half], dtype=np.float64),   # rear-right
            np.array([front_center_x, -track_half], dtype=np.float64), # front-left
            np.array([front_center_x, track_half], dtype=np.float64),  # front-right
        ]

        # rear wheels align with body; front wheels steer
        rear_angle = 0.0
        front_angle = float(state.steering_angle)

        wheels_local = [
            self._wheel_rect(centers[0], rear_angle),
            self._wheel_rect(centers[1], rear_angle),
            self._wheel_rect(centers[2], front_angle),
            self._wheel_rect(centers[3], front_angle),
        ]

        # vehicle-local -> world
        R = self._rot(state.heading)
        t = np.array([state.x, state.y], dtype=np.float64)
        wheels_world = [(w @ R.T) + t for w in wheels_local]
        return wheels_world

    # ------------------------------------------------------------------
    # Drift visualization helpers
    # ------------------------------------------------------------------

    def _kinematic_beta(self, steering_angle: float) -> float:
        """
        Estimate kinematic slip angle beta for drift visualization.

        Parameters
        ----------
        steering_angle : float
            Steering angle (rad).

        Returns
        -------
        float
            Slip angle beta (rad).
        """
        # beta = arctan(lr/L * tan(delta))
        r = self._config.rear_axle_ratio
        return float(np.arctan(r * np.tan(float(steering_angle))))

    def _draw_drift_arc(
        self,
        surface: pygame.Surface,
        origin_screen: Tuple[int, int],
        heading_angle: float,
        velocity_angle: float,
        radius_px: int = 24,
        width: int = 2,
    ) -> None:
        """
        Draw a simple drift arc between heading and velocity directions.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface.
        origin_screen : tuple[int, int]
            Arc center (pixels).
        heading_angle : float
            Heading direction angle (rad) in world coordinates.
        velocity_angle : float
            Velocity direction angle (rad) in world coordinates.
        radius_px : int
            Arc radius in pixels.
        width : int
            Arc line width in pixels.
        """
        # pygame arc angles are in radians and in screen coords, but we render
        # using a symmetric visual approximation: flip sign for y inversion.
        # World angle theta -> screen angle approx -theta
        a0 = -heading_angle
        a1 = -velocity_angle

        # normalize to shortest direction
        def wrap_pi(a: float) -> float:
            return (a + np.pi) % (2.0 * np.pi) - np.pi

        d = wrap_pi(a1 - a0)
        a1 = a0 + d

        rect = pygame.Rect(
            origin_screen[0] - radius_px,
            origin_screen[1] - radius_px,
            2 * radius_px,
            2 * radius_px,
        )

        pygame.draw.arc(surface, self._config.drift_arc_color, rect, a0, a1, width)

    # ------------------------------------------------------------------
    # Public draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface, state: State) -> None:
        """
        Draw vehicle body, wheels, heading, velocity and drift visualization.

        Parameters
        ----------
        surface : pygame.Surface
            Target surface.
        state : State
            Current vehicle state snapshot.
        """
        # Body polygon
        body = self._vehicle_body_polygon(state)
        pygame.draw.polygon(surface, self._config.body_color, self._world_to_screen_points(body))

        # Wheels
        if self._config.show_wheels:
            for wheel in self._wheel_polygons_world(state):
                pygame.draw.polygon(surface, self._config.wheel_color, self._world_to_screen_points(wheel))

        # Origin in screen coords (rear axle point)
        origin_screen = self._world_to_screen_points(np.array([[state.x, state.y]], dtype=np.float64))[0]

        # Heading vector
        if self._config.show_heading:
            L = self._config.body_length
            heading_end = np.array(
                [state.x + L * np.cos(state.heading), state.y + L * np.sin(state.heading)],
                dtype=np.float64,
            )
            p0, p1 = self._world_to_screen_points(np.array([[state.x, state.y], heading_end], dtype=np.float64))
            pygame.draw.line(surface, self._config.heading_color, p0, p1, 2)

        # Velocity direction + drift angle
        if self._config.show_velocity or self._config.show_drift:
            beta = self._kinematic_beta(state.steering_angle)
            vel_dir = float(state.heading + beta)

            # Velocity vector line (length proportional to speed)
            if self._config.show_velocity:
                vlen = max(0.0, float(state.velocity))
                vlen = min(vlen, 25.0)  # cap for readability
                vel_end = np.array(
                    [state.x + vlen * np.cos(vel_dir), state.y + vlen * np.sin(vel_dir)],
                    dtype=np.float64,
                )
                p0, p1 = self._world_to_screen_points(np.array([[state.x, state.y], vel_end], dtype=np.float64))
                pygame.draw.line(surface, self._config.velocity_color, p0, p1, 2)

            # Drift arc between heading and velocity direction
            if self._config.show_drift:
                self._draw_drift_arc(surface, origin_screen, float(state.heading), vel_dir, radius_px=26, width=2)
