"""
rc_racer.main_realtime

Realtime demo entrypoint for RC Racer.

Architecture Rules
------------------
- Simulation loop is authoritative.
- Fixed timestep.
- GUI is passive and never steps environment.
- Deterministic given seed.
"""

from __future__ import annotations

import threading

from rc_racer.core.track_factory import TrackFactory
from rc_racer.core.state import State
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig
from rc_racer.simulation.runner_realtime import (
    RealtimeRunner,
    RunnerConfig,
    SyncControllerProvider,
)
from rc_racer.gui.app import App, AppConfig
from rc_racer.controllers.controllers.pid_controller import PIDLineFollower, PIDConfig


# ================================================================
# ENVIRONMENT BUILDER
# ================================================================


def build_environment() -> tuple[Environment, object]:
    """
    Construct deterministic environment and track.
    """
    track = TrackFactory.create("f1_like_closed")
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
        config=TerminationConfig(
            max_steps=10_000,
            allow_reverse=False,
        ),
    )

    env = Environment(
        track=track,
        vehicle_model=vehicle_model,
        collision_checker=collision_checker,
        reward_system=reward_system,
        termination_condition=termination,
        config=EnvironmentConfig(dt=0.02),
    )

    return env, track


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    """
    Launch realtime demo with GUI.
    """
    env, track = build_environment()

    controller = PIDLineFollower(
        track=track,
        config=PIDConfig(
            kp_lat=6.775,
            ki_lat=1.907,
            kd_lat=3.714,
            kp_head=1.353,
            ki_head=2.432,
            kd_head=2.355,
            kp_speed=1.459,
            ki_speed=0.972,
            kd_speed=3.853,
            target_velocity=10.0,
        ),
    )

    provider = SyncControllerProvider(controller)

    app = App(
        track=track,
        config=AppConfig(
            width=1200,
            height=700,
            pixels_per_meter=6.0,
            show_debug=True
        ),
    )

    lap_time = 0.0
    total_score = 0.0

    def on_step(state: State, reward: float, done: bool, info: dict) -> None:
        nonlocal lap_time, total_score

        lap_time += env.dt
        total_score += reward

        debug_vals = controller.debug_values

        app.update_state(
            state,
            score=total_score,
            lap_time=lap_time,
            debug_values=debug_vals,
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

    sim_thread = threading.Thread(
        target=runner.run,
        kwargs={"seed": 42},
        daemon=True,
    )

    sim_thread.start()
    app.run()

    runner.stop()
    sim_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
