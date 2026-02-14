"""
train_pid_cmaes_single_track.py

CMA-ES training script for the multi-loop PID controller
using the GenomeInterface system.

Architecture
------------
- Evolution layer: CMAESTrainer
- Agents layer: PIDConfig implements GenomeInterface
- Environment layer: created per evaluation
- Parallelism outside environment

Features
--------
- Single deterministic track
- Multi-seed robustness
- Parallel evaluation
- Genome serialization
- Deterministic given base seed
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray

from rc_racer.controllers.controllers.pid_controller import PIDLineFollower, PIDConfig
from rc_racer.controllers.trainers.cmaes import (
    CMAESTrainer,
    CMAESTrainerConfig,
)
from rc_racer.simulation.runner_batch import BatchRunner, BatchRunnerConfig
from rc_racer.core.track import Track
from rc_racer.core.track_factory import TrackFactory
from rc_racer.core.vehicle_factory import VehicleFactory
from rc_racer.environment.environment import Environment, EnvironmentConfig
from rc_racer.environment.collision import CollisionChecker
from rc_racer.environment.reward import RewardSystem, RewardConfig
from rc_racer.environment.termination import TerminationCondition, TerminationConfig


FloatArray = NDArray[np.float64]


# ================================================================
# Environment Factory
# ================================================================


def make_environment(track: Track) -> Environment:
    """
    Create fresh deterministic environment instance.

    Parameters
    ----------
    track : Track

    Returns
    -------
    Environment
    """

    vehicle_model = VehicleFactory.create_model("default")

    collision_checker = CollisionChecker(track)

    reward_system = RewardSystem(
        RewardConfig(
            progress_weight=1.0,
            off_track_penalty=100.0,
            time_penalty=0.0,
            finish_bonus=300.0,
        )
    )

    termination = TerminationCondition(
        total_track_length=track.total_length,
        config=TerminationConfig(max_steps=2400),
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
# Fitness Evaluation (Picklable)
# ================================================================


def evaluate_genome_vector(
    z: FloatArray,
    track: Track,
    seeds: List[int],
) -> float:
    """
    Decode genome vector and evaluate across seeds.

    Parameters
    ----------
    z : ndarray
        Raw genome vector.
    track : Track
    seeds : list[int]

    Returns
    -------
    float
        Mean fitness across seeds.
    """

    config = PIDConfig.decode(z)

    fitnesses: List[float] = []

    for seed in seeds:
        env = make_environment(track)

        controller = PIDLineFollower(
            track=track,
            config=config,
        )

        runner = BatchRunner(
            env=env,
            controller=controller,
            config=BatchRunnerConfig(max_steps=2400),
        )

        result = runner.run(seed=seed)
        fitnesses.append(result.total_reward)

    return float(np.mean(np.asarray(fitnesses, dtype=np.float64)))


# ================================================================
# Parallel Mapping Wrapper (Picklable)
# ================================================================


def parallel_map(
    fn,
    candidates: List[FloatArray],
    track: Track,
    seeds: List[int],
) -> List[float]:
    """
    Parallel fitness evaluation.

    Parameters
    ----------
    fn : Callable
    candidates : list[np.ndarray]
    track : Track
    seeds : list[int]

    Returns
    -------
    list[float]
    """

    with ProcessPoolExecutor() as ex:
        futures = [
            ex.submit(fn, z)
            for z in candidates
        ]
        return [f.result() for f in futures]


# ================================================================
# Main Training Loop
# ================================================================


def main() -> None:
    """
    Run CMA-ES PID training.
    """

    out_dir = Path("runs/train_pid_cmaes_single_track")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = 1234

    # ------------------------------------------------------------
    # Track (immutable, safe to share)
    # ------------------------------------------------------------
    track = TrackFactory.create("curved_s_track")

    # Multi-seed robustness
    seeds = [0, 1, 2, 3]

    # ------------------------------------------------------------
    # CMA-ES Setup
    # ------------------------------------------------------------
    trainer = CMAESTrainer(
        CMAESTrainerConfig(
            genome_size=PIDConfig.genome_size(),
            sigma=1.2,
            population_size=40,
            generations=100,
            seed=base_seed,
        )
    )

    # ------------------------------------------------------------
    # Picklable Evaluation Functions
    # ------------------------------------------------------------

    eval_fn = partial(
        evaluate_genome_vector,
        track=track,
        seeds=seeds,
    )

    map_fn = partial(
        parallel_map,
        track=track,
        seeds=seeds,
    )

    # ------------------------------------------------------------
    # Run Optimization
    # ------------------------------------------------------------

    result = trainer.optimize(
        evaluate_fn=eval_fn,
        map_fn=map_fn,
        verbose=True,
    )

    best_config = PIDConfig.decode(result.best_genome)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Best fitness:", result.best_fitness)
    print("Best PID Config:")
    print(best_config)
    print("=" * 60)

    # Save best genome
    with open(out_dir / "best_genome.json", "w") as f:
        f.write(best_config.to_json(indent=2))


# ================================================================
# Entry Point
# ================================================================


if __name__ == "__main__":
    main()
