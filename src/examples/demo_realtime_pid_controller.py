"""
demo_realtime_runner_pid_curved.py

Realtime demonstration using RealtimeRunner with:

- Smooth curved S-track
- PID centerline-following controller
- Deterministic Environment stepping
- Track boundary collision
- Authoritative simulation clock via RealtimeRunner
- Passive GUI rendering via on_step callback

Architecture
------------
Simulation loop is authoritative (RealtimeRunner).
Environment remains synchronous and deterministic.
GUI is passive.
Controller does not control timestep.

Layer: SIMULATION (demo usage only)
"""

from __future__ import annotations

import pygame
import numpy as np
from typing import Tuple

from rc_racer.core.track import Track
from rc_racer.core.vehicle_model import VehicleModel, VehicleParams
from rc_racer.env.environment import Environment, EnvironmentConfig
from rc_racer.env.collision import CollisionChecker, CollisionConfig
from rc_racer.env.reward import RewardSystem, RewardConfig
from rc_racer.env.termination import TerminationCondition, TerminationConfig
from rc_racer.gui.track_view import TrackView, TrackViewConfig
from rc_racer.gui.agent_view import PygameAgentView, AgentViewConfig
from rc_racer.agents.pid_controller import PIDLineFollower, PIDConfig
from rc_racer.simulation.runner_realtime import (
    RealtimeRunner,
    RunnerConfig,
    SyncControllerProvider,
)

Color = Tuple[int, int, int]


# ================================================================
# BUILDERS
# ================================================================


def build_curved_s_track() -> Track:
    """
    Build a smooth S-shaped track.

    Returns
    -------
    Track
    """
    xs = np.linspace(0.0, 120.0, 600)

    # Smooth S-curve using sinusoidal modulation
    ys = (
        8.0 * np.sin(0.08 * xs) +
        4.0 * np.sin(0.18 * xs)
    )

    centerline = np.column_stack((xs, ys)).astype(np.float64)

    return Track(
        centerline=centerline,
        width=10.0,
    )


def build_vehicle_model() -> VehicleModel:
    """
    Create deterministic kinematic bicycle model.

    Returns
    -------
    VehicleModel
    """
    params = VehicleParams(
        wheelbase=2.5,
        rear_axle_ratio=0.5,
        max_steering_angle=0.6,
        max_steering_rate=1.5,
        max_acceleration=6.0,
        max_velocity=25.0,
        mu=1.0,
        g=9.81,
        a_lat_max=7.0,
        mass=1200.0,
        c_rr=0.015,
        c_d_a_over_m=0.0004,
    )
    return VehicleModel(params)


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    pygame.display.set_caption("RC Racer - PID Curved Track Demo")
    clock = pygame.time.Clock()

    # ------------------------------------------------------------
    # Core components
    # ------------------------------------------------------------

    track = build_curved_s_track()
    vehicle_model = build_vehicle_model()

    collision_checker = CollisionChecker(
        track,
        CollisionConfig(use_footprint=False),
    )

    reward_system = RewardSystem(
        RewardConfig(
            progress_weight=1.0,
            off_track_penalty=20.0,
            time_penalty=0.01,
            finish_bonus=100.0,
        )
    )

    termination_condition = TerminationCondition(
        total_track_length=track.total_length,
        config=TerminationConfig(max_steps=8000),
    )

    env = Environment(
        track=track,
        vehicle_model=vehicle_model,
        collision_checker=collision_checker,
        reward_system=reward_system,
        termination_condition=termination_condition,
        config=EnvironmentConfig(dt=0.02),
    )

    # ------------------------------------------------------------
    # PID Controller instead of ForwardController
    # ------------------------------------------------------------

    pid_config = PIDConfig(
        kp=2.0,
        ki=0.0,
        kd=0.6,
        target_velocity=10.0,
        speed_kp=2.0,
    )

    controller = PIDLineFollower(track=track, config=pid_config)
    provider = SyncControllerProvider(controller)

    # ------------------------------------------------------------
    # GUI setup (passive)
    # ------------------------------------------------------------

    pixels_per_meter = 6.0
    offset = (100, 400)

    track_view = TrackView(
        track,
        TrackViewConfig(pixels_per_meter=pixels_per_meter),
        screen_offset_px=offset,
    )

    agent_view = PygameAgentView(
        AgentViewConfig(pixels_per_meter=pixels_per_meter),
        screen_offset_px=offset,
    )

    running = True

    # ------------------------------------------------------------
    # on_step callback (passive GUI)
    # ------------------------------------------------------------

    def on_step(state, reward, done, info) -> None:
        nonlocal running

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((25, 25, 25))
        track_view.draw(screen)
        agent_view.draw(screen, state)
        pygame.display.flip()

        clock.tick(60)

        if info.get("collision", False):
            print("Collision detected. Ending demo.")
            pygame.time.wait(1000)
            running = False

        if done:
            running = False

    # ------------------------------------------------------------
    # Realtime runner (authoritative loop)
    # ------------------------------------------------------------

    runner = RealtimeRunner(
        env=env,
        action_provider=provider,
        config=RunnerConfig(
            dt=env.dt,
            target_fps=60.0,
        ),
        on_step=on_step,
    )

    runner.run()

    pygame.quit()


if __name__ == "__main__":
    main()
