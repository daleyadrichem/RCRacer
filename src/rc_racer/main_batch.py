"""
rc_racer.main_batch

Batch execution entry point for deterministic evaluation.

Architecture
------------
- Environment recreated per evaluation.
- No shared mutable state.
- Deterministic given seed.
- Parallel-safe.
"""

from __future__ import annotations

from rc_racer.core.track_factory import TrackFactory
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker, CollisionConfig
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig
from rc_racer.simulation.evaluator import Evaluator, EvaluatorConfig
from rc_racer.agents.base_controller import BaseController
from controllers.controllers.pid_controller import PIDLineFollower, PIDConfig


# ================================================================
# ENVIRONMENT FACTORY
# ================================================================


def make_environment() -> Environment:
    """
    Create deterministic environment instance.
    """
    track = TrackFactory.create("closed_challenging")
    vehicle_model = VehicleFactory.create_model("default")

    collision_checker = CollisionChecker(
        track=track,
        config=CollisionConfig(use_footprint=False),
    )

    reward_system = RewardSystem(
        RewardConfig(
            progress_weight=1.0,
            off_track_penalty=50.0,
            time_penalty=0.01,
            finish_bonus=200.0,
        )
    )

    termination = TerminationCondition(
        total_track_length=track.total_length,
        config=TerminationConfig(
            max_steps=3000,
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
# CONTROLLER FACTORY
# ================================================================


def make_controller(_: None) -> BaseController:
    """
    Create deterministic PID controller.
    """
    track = TrackFactory.create("closed_challenging")

    config = PIDConfig(
        kp_lat=3.0,
        ki_lat=0.0,
        kd_lat=0.5,
        kp_head=3.0,
        ki_head=0.0,
        kd_head=0.5,
        kp_speed=2.0,
        ki_speed=0.0,
        kd_speed=0.0,
        target_velocity=12.0,
    )

    return PIDLineFollower(
        track=track,
        config=config,
    )


# ================================================================
# MAIN
# ================================================================


def main() -> None:
    """
    Run deterministic batch evaluation.
    """
    evaluator = Evaluator[None](
        env_factory=make_environment,
        controller_factory=make_controller,
        config=EvaluatorConfig(max_steps=3000),
    )

    fitness = evaluator.evaluate(
        genome=None,
        seed=42,
    )

    print("=== Batch Evaluation Complete ===")
    print(f"Fitness: {fitness:.3f}")


if __name__ == "__main__":
    main()
    