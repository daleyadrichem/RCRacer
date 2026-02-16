"""
main_synchronous.py

Synchronous deterministic simulation entrypoint.

Supports:
- Live display mode
- Headless video export mode

Architecture Compliance
-----------------------
- Environment remains authoritative and synchronous.
- Controller is called every step.
- No real-time clock usage.
- GUI is passive (render only).
- Deterministic given seed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from rc_racer.gui.app import App, AppConfig

from rc_racer.controllers.controllers.mpcc_controller import (
    MpccController,
    MpccConfig,
)


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class SynchronousConfig:
    """
    Configuration for synchronous simulation.

    Parameters
    ----------
    width : int
        Render width in pixels.
    height : int
        Render height in pixels.
    max_steps : int
        Maximum simulation steps.
    fps : int
        Target FPS for display mode or video export.
    output_path : str
        Output video file path (used in video mode).
    """

    width: int = 1920
    height: int = 1200
    max_steps: int = 10000
    fps: int = 60
    output_path: str = "simulation.mp4"


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


def main(mode: Literal["display", "video"]) -> None:
    """
    Run deterministic synchronous simulation.

    Parameters
    ----------
    mode : {"display", "video"}
        Execution mode.
    """

    cfg = SynchronousConfig()

    track = TrackFactory.create("f1_like_closed")
    env = make_environment(track)

    controller = MpccController(
        track=track,
        vehicle_params=env._vehicle_model._p,
        config=MpccConfig(
            dt=env.dt,
            v_ref=30.0,
            w_v_min=5000.0,
        ),
    )

    state: State = env.reset(seed=42)
    controller.reset()

    pygame.init()

    if mode == "display":
        writer = None
    else:
        writer = imageio.get_writer(Path(cfg.output_path), fps=cfg.fps)

    app = App(
        track=track,
        config=AppConfig(
            width=cfg.width,
            height=cfg.height,
            pixels_per_meter=12.0,  # or whatever you want
            window_title="RC Racer - Synchronous Mode",
        ),
    )

    total_score = 0.0
    lap_time = 0.0

    loop_iter = range(cfg.max_steps)
    if mode == "video":
        loop_iter = tqdm(loop_iter, desc="Simulating", unit="step")

    current_action = (0.0, 0.0)
    predicted_path = None

    for step in loop_iter:
        if step % controller._cfg.control_block_steps == 0:
            current_action = controller.compute_action(state)
            predicted_path = controller.get_last_predicted_path()
            
        next_state, reward, done, _ = env.step(current_action)

        total_score += reward
        lap_time += env.dt

        app.update_state(
            next_state,
            score=total_score,
            lap_time=lap_time,
            predicted_path=predicted_path
        )

        app._render()


        if mode == "display":
            pygame.display.flip()
            app._clock.tick(cfg.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
        else:
            frame = pygame.surfarray.array3d(app._screen)
            frame = np.transpose(frame, (1, 0, 2))
            writer.append_data(frame)

        state = next_state

        if done:
            break

    if writer is not None:
        writer.close()
        print(f"Video saved to: {Path(cfg.output_path).resolve()}")

    pygame.quit()


# ================================================================
# ENTRYPOINT
# ================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RC Racer synchronous simulation."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="display",
        choices=["display", "video"],
        help="Execution mode: display live window or export video.",
    )

    args = parser.parse_args()
    main(mode=args.mode)
