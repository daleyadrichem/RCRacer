"""
demo_agent_view_random.py

Pygame demo that visualizes a vehicle with wheel + drift visualization
using random (seeded) actions and deterministic VehicleModel stepping.

Run
---
python -m scripts.demo_agent_view_random

Notes
-----
- Randomness is seeded for reproducibility.
- This is a demo script (not core simulation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from rc_racer.core.state import State
from rc_racer.core.vehicle_model import VehicleModel, VehicleParams
from rc_racer.gui.agent_view import AgentViewConfig, PygameAgentView


@dataclass(frozen=True)
class DemoConfig:
    """
    Demo configuration.

    Parameters
    ----------
    width_px : int
        Screen width in pixels.
    height_px : int
        Screen height in pixels.
    fps : int
        Rendering FPS (fixed dt = 1/fps).
    seed : int
        RNG seed for reproducible random actions.
    world_wrap_m : float
        Wrap boundary in meters (vehicle position wraps within [-wrap, +wrap]).
    """

    width_px: int = 1000
    height_px: int = 700
    fps: int = 60
    seed: int = 0
    world_wrap_m: float = 40.0


def main() -> None:
    cfg = DemoConfig()

    pygame.init()
    screen = pygame.display.set_mode((cfg.width_px, cfg.height_px))
    pygame.display.set_caption("RCRacer AgentView Demo (Wheels + Drift)")
    clock = pygame.time.Clock()

    # Renderer config
    ppm = 12.0
    view_cfg = AgentViewConfig(
        pixels_per_meter=ppm,
        body_length=4.0,
        body_width=2.0,
        wheelbase=2.5,
        rear_axle_ratio=0.5,
        show_wheels=True,
        show_heading=True,
        show_velocity=True,
        show_drift=True,
    )

    # Place origin at screen center
    view = PygameAgentView(view_cfg, screen_offset_px=(cfg.width_px // 2, cfg.height_px // 2))

    # Vehicle model (deterministic)
    params = VehicleParams(
        wheelbase=2.5,
        rear_axle_ratio=0.5,
        max_steering_angle=np.deg2rad(30),
        max_steering_rate=np.deg2rad(180),
        max_acceleration=5.0,
        max_velocity=35.0,
        mu=1.0,
        g=9.81,
        a_lat_max=8.0,
        mass=1200.0,
        c_rr=0.015,
        c_d_a_over_m=0.0005,
    )
    model = VehicleModel(params)

    # Initial state near origin
    state = State(
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity=8.0,
        steering_angle=0.0,
        progress_s=0.0,
    )

    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / float(cfg.fps)

    running = True
    while running:
        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Random-but-seeded actions (accel, steering_rate)
        # Slightly correlated behavior: use small noise and occasional bias.
        accel_cmd = float(rng.normal(loc=0.0, scale=1.2))
        steer_rate_cmd = float(rng.normal(loc=0.0, scale=0.9))

        # Clamp to keep it visually nice
        accel_cmd = float(np.clip(accel_cmd, -3.0, 3.0))
        steer_rate_cmd = float(np.clip(steer_rate_cmd, -1.5, 1.5))

        state = model.step(state, (accel_cmd, steer_rate_cmd), dt)

        # Wrap position so we always stay in view without a camera system
        wrap = cfg.world_wrap_m
        x = state.x
        y = state.y
        if x > wrap:
            x -= 2.0 * wrap
        elif x < -wrap:
            x += 2.0 * wrap
        if y > wrap:
            y -= 2.0 * wrap
        elif y < -wrap:
            y += 2.0 * wrap
        if x != state.x or y != state.y:
            state = state.copy_with(x=x, y=y)

        # Render
        screen.fill((235, 235, 235))
        view.draw(screen, state)
        pygame.display.flip()
        clock.tick(cfg.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
