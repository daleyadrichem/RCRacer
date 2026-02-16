"""
main_offline_video.py

Offline (non-realtime) simulation entrypoint with tqdm progress bar.

This file:
- Runs controller synchronously (no threading, no sleeping)
- Advances deterministic environment step-by-step
- Renders each step using passive GUI components
- Exports a video file
- Displays tqdm progress bar

Architecture Compliance
-----------------------
- Environment remains authoritative and synchronous.
- Controller is called every step.
- No real-time clock usage.
- GUI is passive (render only).
- Deterministic given seed.

References
----------
Architecture specification: :contentReference[oaicite:2]{index=2}
Batch runner pattern: :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame
import imageio.v2 as imageio
from tqdm import tqdm

from rc_racer.core.track import Track
from rc_racer.core.track_factory import TrackFactory
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig
from rc_racer.core.state import State
from rc_racer.gui.track_view import TrackView
from rc_racer.gui.agent_view import PygameAgentView
from rc_racer.gui.dashboard_view import (
    PygameDashboardView,
    make_dashboard_theme,
)
from rc_racer.controllers.controllers.mpcc_controller import MpccController, MpccConfig
from rc_racer.controllers.controllers.pid_controller import PIDConfig, PIDLineFollower


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class OfflineVideoConfig:
    """
    Configuration for offline video generation.
    """

    width: int = 1920
    height: int = 1200
    pixels_per_meter: float = 5.0
    max_steps: int = 1000
    output_path: str = "simulation.mp4"
    fps: int = 60


# ================================================================
# ENVIRONMENT FACTORY
# ================================================================


def make_environment(track: Track) -> Environment:
    """
    Create deterministic environment instance.
    """
    vehicle_model = VehicleFactory.create_model("default")

    collision_checker = CollisionChecker(track)

    reward_system = RewardSystem(
        RewardConfig(
            progress_weight=1.0,
            off_track_penalty=50.0,
            time_penalty=0.01,
            finish_bonus=100.0,
        )
    )

    termination = TerminationCondition(
        total_track_length=track.total_length,
        config=TerminationConfig(max_steps=10_000),
    )

    return Environment(
        track=track,
        vehicle_model=vehicle_model,
        collision_checker=collision_checker,
        reward_system=reward_system,
        termination_condition=termination,
        config=EnvironmentConfig(dt=0.05),
    )


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    """
    Run offline simulation and export video with tqdm progress bar.
    """

    cfg = OfflineVideoConfig()

    # ------------------------------------------------------------
    # Build simulation
    # ------------------------------------------------------------

    track = TrackFactory.create("f1_like_closed")
    env = make_environment(track)

    controller = MpccController(
        track=track,
        vehicle_params=env._vehicle_model._p,
        config=MpccConfig(
            dt=env.dt,    
            v_ref=15.0,
            w_v_min=5000.0,      
        ),
    )    
    # controller = PIDLineFollower(
    #     track=track,
    #     config=PIDConfig(
    #         kp_lat=2.699,
    #         ki_lat=0.019,
    #         kd_lat=1.429,
    #         kp_head=15.853,
    #         ki_head=0.005,
    #         kd_head=4.508,
    #         kp_speed=7.499,
    #         ki_speed=0.189,
    #         kd_speed=0.439,
    #         target_velocity=8.804,
    #     ),
    # )

    state: State = env.reset(seed=42)
    controller.reset()

    # ------------------------------------------------------------
    # Setup pygame (headless surface)
    # ------------------------------------------------------------

    pygame.init()
    surface = pygame.Surface((cfg.width, cfg.height))

    offset = (cfg.width // 4, cfg.height // 2)

    track_view = TrackView(
        track,
        screen_offset_px=offset,
    )

    agent_view = PygameAgentView(
        screen_offset_px=offset,
    )

    dashboard_view = PygameDashboardView(
        make_dashboard_theme(
            theme="dark",
            panel_position_px=(20, 20),
        )
    )

    # ------------------------------------------------------------
    # Video writer
    # ------------------------------------------------------------

    output_path = Path(cfg.output_path)
    writer = imageio.get_writer(output_path, fps=cfg.fps)

    total_score = 0.0
    lap_time = 0.0

    # ------------------------------------------------------------
    # Authoritative simulation loop
    # ------------------------------------------------------------

    with tqdm(
        total=cfg.max_steps,
        desc="Simulating",
        unit="step",
    ) as pbar:

        for _ in range(cfg.max_steps):

            action = controller.compute_action(state)

            if _ == 0:
                print("mpcc action step0:", action, controller.debug_values)

            next_state, reward, done, _info = env.step(action)

            total_score += reward
            lap_time += env.dt

            # ----------------------------------------------------
            # Render frame
            # ----------------------------------------------------

            surface.fill((30, 30, 30))

            track_view.draw(surface)
            agent_view.draw(surface, next_state)
            dashboard_view.draw(
                surface,
                state=next_state,
                score=total_score,
                lap_time=lap_time,
                fps=float(cfg.fps),
            )

            frame = pygame.surfarray.array3d(surface)
            frame = np.transpose(frame, (1, 0, 2))
            writer.append_data(frame)

            state = next_state

            pbar.update(1)

            if done:
                pbar.set_postfix({"status": "terminated"})
                break

    writer.close()
    pygame.quit()

    print(f"\nVideo saved to: {output_path.resolve()}")


# ================================================================
# ENTRYPOINT
# ================================================================


if __name__ == "__main__":
    main()
