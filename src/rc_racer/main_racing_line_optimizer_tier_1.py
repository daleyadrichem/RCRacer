"""
main_racing_line_offline.py

Offline racing-line optimizer entrypoint.

- Computes an ideal racing line + global speed profile + dt actions offline
- Replays the resulting lap in display or video mode
- Overlays the ideal line using a passive view
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

from rc_racer.core.track_factory import TrackFactory
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.gui.app import App, AppConfig
from rc_racer.gui.race_line_view import PygameRacingLineView, RacingLineViewConfig

from rc_racer.controllers.controllers.racing_line_optimizer_tier_1 import (
    Tier1RacingLineOptimizer,
    Tier1RacingLineOptimizerConfig,
    SpeedProfileConfig,
)


@dataclass(frozen=True)
class OfflineRacingLineMainConfig:
    """
    Configuration for offline racing-line replay.

    Parameters
    ----------
    width : int
        Render width.
    height : int
        Render height.
    fps : int
        Playback FPS (display) or export FPS (video).
    output_path : str
        Output video path.
    pixels_per_meter : float
        World scale.
    """

    width: int = 1920
    height: int = 1200
    fps: int = 60
    output_path: str = "racing_line_offline.mp4"
    pixels_per_meter: float = 12.0


def main(mode: Literal["display", "video"]) -> None:
    cfg = OfflineRacingLineMainConfig()

    print("Creating track...")
    track = TrackFactory.create("f1_like_closed")

    print("Creating vehicle model...")
    vehicle_model = VehicleFactory.create_model("default")
    vehicle_params = vehicle_model._p

    print("Running offline racing-line optimizer...")
    optimizer = Tier1RacingLineOptimizer(
        line_cfg=Tier1RacingLineOptimizerConfig(
            ds=0.25,
            margin=0.25,
            iterations=500,
            beta_outside=0.95,
            smooth_alpha=0.28,
            attract_gamma=0.08,
        ),
        speed_cfg=SpeedProfileConfig(
            dt=0.05,
            v_start=0.0,
            v_end=0.0,
            speed_kp=1.0,
            steer_kp=6.0,
        ),
        show_progress=True,
    )

    plan = optimizer.optimize(
        track=track,
        vehicle_model=vehicle_model,
        vehicle_params=vehicle_params,
    )

    print(f"Optimization complete.")
    print(f"Replay steps: {len(plan.states)}")
    print(f"Estimated lap time: {plan.times_s[-1]:.2f} s")

    # ------------------ GUI / Video ------------------
    pygame.init()

    writer = None if mode == "display" else imageio.get_writer(Path(cfg.output_path), fps=cfg.fps)

    app = App(
        track=track,
        config=AppConfig(
            width=cfg.width,
            height=cfg.height,
            pixels_per_meter=cfg.pixels_per_meter,
            window_title="RC Racer - Offline Racing Line",
        ),
    )

    # Ideal line overlay view (passive)
    line_view = PygameRacingLineView(
        path_points=plan.path_points,
        config=RacingLineViewConfig(
            pixels_per_meter=cfg.pixels_per_meter,
            color=(255, 80, 80),
            width_px=4,
        ),
        screen_offset_px=app._config.screen_offset_px,  # consistent transform
    )

    total_steps = len(plan.states)
    loop_iter = range(total_steps)
    if mode == "video":
        loop_iter = tqdm(loop_iter, desc="Rendering", unit="frame")

    for i in loop_iter:
        st = plan.states[i]
        app.update_state(st, score=0.0, lap_time=float(plan.times_s[i]))

        # Custom render (same pattern as your synchronous main uses private members)
        app._screen.fill(app._config.background_color)
        app._track_view.draw(app._screen)
        line_view.draw(app._screen)  # <-- ideal line overlay
        if app._current_state is not None:
            app._agent_view.draw(app._screen, app._current_state)
            app._dashboard.draw(
                app._screen,
                state=app._current_state,
                score=0.0,
                lap_time=float(plan.times_s[i]),
                fps=app._clock.get_fps(),
            )

        if mode == "display":
            pygame.display.flip()
            app._clock.tick(cfg.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
        else:
            frame = pygame.surfarray.array3d(app._screen)
            frame = np.transpose(frame, (1, 0, 2))
            writer.append_data(frame)

    if writer is not None:
        writer.close()
        print(f"Video saved to: {Path(cfg.output_path).resolve()}")

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline racing line optimizer + replay.")
    parser.add_argument(
        "--mode",
        type=str,
        default="display",
        choices=["display", "video"],
        help="display = live window, video = export mp4",
    )
    args = parser.parse_args()
    main(mode=args.mode)
