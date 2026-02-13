"""
main_realtime.py

Realtime demo entrypoint for RC Racer.

This file wires together:

- Core simulation (Track, VehicleModel, Environment)
- PID controller
- RealtimeRunner (authoritative simulation loop)
- Passive GUI (App)

Architecture Rules
------------------
- Simulation loop is authoritative.
- Fixed timestep.
- GUI is passive and never steps environment.
- No hidden randomness.
- Deterministic given seed.

See architecture specification for details.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from rc_racer.core.track import Track
from rc_racer.core.vehicle_model import VehicleModel, VehicleParams
from rc_racer.core.state import State
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig
from rc_racer.agents.pid_controller import PIDLineFollower, PIDConfig
from rc_racer.agents.mpcc_controller import MpccController, MpccConfig
from rc_racer.simulation.runner_realtime import (
    RealtimeRunner,
    RunnerConfig,
    SyncControllerProvider,
)
from rc_racer.gui.app import App, AppConfig


# ================================================================
# TRACK
# ================================================================


def build_curved_s_track() -> Track:
    """
    Build the same curved S track used by the GUI demo.

    Returns
    -------
    Track
    """
    xs = np.linspace(0.0, 120.0, 600)
    ys = 8.0 * np.sin(0.08 * xs) + 4.0 * np.sin(0.18 * xs)
    centerline = np.column_stack((xs, ys)).astype(np.float64)

    return Track(centerline=centerline, width=10.0)


# ================================================================
# SIMULATION SETUP
# ================================================================


def build_environment(track: Track) -> Environment:
    """
    Construct deterministic racing environment.

    Parameters
    ----------
    track : Track

    Returns
    -------
    Environment
    """
    vehicle_model = VehicleModel(
        VehicleParams(
            wheelbase=2.6,
            rear_axle_ratio=0.5,
            max_steering_angle=0.6,
            max_steering_rate=1.5,
            max_acceleration=6.0,
            max_velocity=20.0,
            mu=1.2,
            g=9.81,
            a_lat_max=12.0,
            mass=1200.0,
            c_rr=0.015,
            c_d_a_over_m=0.0025,
        )
    )

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
        config=TerminationConfig(
            max_steps=10_000,
            allow_reverse=False,
        ),
    )

    return Environment(
        track=track,
        vehicle_model=vehicle_model,
        collision_checker=collision_checker,
        reward_system=reward_system,
        termination_condition=termination,
        config=EnvironmentConfig(dt=0.02),
    )


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    """
    Launch realtime demo with GUI.
    """
    # ------------------------------------------------------------
    # Build track & environment
    # ------------------------------------------------------------

    track = build_curved_s_track()
    env = build_environment(track)

    # ------------------------------------------------------------
    # Controller
    # ------------------------------------------------------------

    # controller = PIDLineFollower(
    #     track=track,
    #     config=PIDConfig(
    #         kp=1.0,
    #         kd=0.0
    #     ),
    # )

    controller = MpccController(track=track)
    provider = SyncControllerProvider(controller)

    # ------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------

    app = App(
        AppConfig(
            width=1200,
            height=700,
            pixels_per_meter=6.0,
        )
    )

    # ------------------------------------------------------------
    # Realtime runner (authoritative loop)
    # ------------------------------------------------------------

    lap_time_accumulator = 0.0
    total_score = 0.0

    def on_step(state: State, reward: float, done: bool, info: dict) -> None:
        """
        Callback from authoritative simulation loop.

        This is the ONLY place GUI receives state.
        """
        nonlocal lap_time_accumulator, total_score

        lap_time_accumulator += env.dt
        total_score += reward

        app.update_state(
            state,
            score=total_score,
            lap_time=lap_time_accumulator,
        )

        if done:
            runner.stop()

    runner = RealtimeRunner(
        env=env,
        action_provider=provider,
        config=RunnerConfig(
            dt=env.dt,
            target_fps=60.0,
        ),
        on_step=on_step,
    )

    # ------------------------------------------------------------
    # Run simulation in background thread
    # ------------------------------------------------------------

    sim_thread = threading.Thread(
        target=runner.run,
        kwargs={"seed": 42},
        daemon=True,
    )

    sim_thread.start()

    # ------------------------------------------------------------
    # Run GUI (main thread)
    # ------------------------------------------------------------

    app.run()

    # When GUI exits:
    runner.stop()
    sim_thread.join(timeout=2.0)


# ================================================================
# ENTRYPOINT
# ================================================================


if __name__ == "__main__":
    main()
