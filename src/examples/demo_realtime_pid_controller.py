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

from rc_racer.core.track_factory import TrackFactory
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker, CollisionConfig
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig
from rc_racer.gui.track_view import TrackView, TrackViewConfig
from rc_racer.gui.agent_view import PygameAgentView, AgentViewConfig
from rc_racer.gui.debug_bar_view import DebugBarView
from rc_racer.controllers.controllers.pid_controller import PIDLineFollower, PIDConfig
from rc_racer.simulation.runner_realtime import (
    RealtimeRunner,
    RunnerConfig,
    SyncControllerProvider,
)

Color = Tuple[int, int, int]

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

    track = TrackFactory.create("f1_like_closed")
    vehicle_model = VehicleFactory.create_model("default")

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
        # Lateral (meters -> steering rate)
        kp_lat=2.5,
        ki_lat=0.0,
        kd_lat=0.8,

        # Heading (radians -> steering rate)
        kp_head=3.0,
        ki_head=0.0,
        kd_head=0.4,

        # Speed (m/s -> acceleration)
        kp_speed=2.0,
        ki_speed=0.2,
        kd_speed=0.0,

        target_velocity=8.0,
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

    debug_view = DebugBarView()
            
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
        acceleration, u_lat, u_head = controller.debug_values
        debug_view.draw(screen, acceleration, u_lat, u_head)

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
