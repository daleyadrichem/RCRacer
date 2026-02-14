"""
rc_racer.simulation.evaluator

Deterministic evaluation utility for batch / parallel training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from rc_racer.agents.base_controller import BaseController
from rc_racer.environment.environment import Environment
from rc_racer.simulation.runner_batch import (
    BatchRunner,
    BatchRunnerConfig,
)


# ================================================================
# GENERIC TYPES
# ================================================================

GenomeT = TypeVar("GenomeT")


# ================================================================
# CONFIGURATION
# ================================================================


@dataclass(frozen=True)
class EvaluatorConfig:
    """
    Configuration for deterministic evaluation.

    Parameters
    ----------
    max_steps : int
        Maximum episode length.
    seed_offset : int
        Optional deterministic offset added to evaluation seed.
        Useful when evaluating a population deterministically.
    """

    max_steps: int
    seed_offset: int = 0


# ================================================================
# EVALUATOR
# ================================================================


class Evaluator(Generic[GenomeT]):
    """
    Deterministic evaluation wrapper.

    Usage
    -----
    evaluator = Evaluator(
        env_factory=make_env,
        controller_factory=make_controller,
        config=EvaluatorConfig(max_steps=2000),
    )

    fitness = evaluator.evaluate(genome)

    Design Goals
    ------------
    - Safe for multiprocessing
    - Deterministic given seed
    - No global state
    - Environment fully recreated per evaluation
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[], Environment],
        controller_factory: Callable[[GenomeT], BaseController],
        config: EvaluatorConfig,
    ) -> None:
        """
        Initialize evaluator.

        Parameters
        ----------
        env_factory : Callable[[], Environment]
            Factory that creates a fresh deterministic environment.
            Must not reuse instances across calls.
        controller_factory : Callable[[GenomeT], BaseController]
            Factory that creates a controller from a genome.
        config : EvaluatorConfig
            Evaluation configuration.
        """
        self._env_factory: Callable[[], Environment] = env_factory
        self._controller_factory: Callable[[GenomeT], BaseController] = (
            controller_factory
        )
        self._config: EvaluatorConfig = config

    # ------------------------------------------------------------

    @property
    def config(self) -> EvaluatorConfig:
        """
        Return evaluator configuration.

        Returns
        -------
        EvaluatorConfig
        """
        return self._config

    # ------------------------------------------------------------

    def evaluate(
        self,
        genome: GenomeT,
        *,
        seed: int | None = None,
    ) -> float:
        """
        Evaluate a genome and return scalar fitness.

        Parameters
        ----------
        genome : GenomeT
            Genome representation (controller parameters, NN weights, etc.).
        seed : int | None
            Optional evaluation seed.

        Returns
        -------
        float
            Deterministic fitness score.
        """
        # --------------------------------------------------------
        # Create fresh environment (no shared state)
        # --------------------------------------------------------
        env: Environment = self._env_factory()

        # --------------------------------------------------------
        # Create controller from genome
        # --------------------------------------------------------
        controller: BaseController = self._controller_factory(genome)

        # --------------------------------------------------------
        # Run batch episode
        # --------------------------------------------------------
        runner = BatchRunner(
            env=env,
            controller=controller,
            config=BatchRunnerConfig(
                max_steps=self._config.max_steps,
            ),
        )

        final_seed = (
            None
            if seed is None
            else int(seed) + int(self._config.seed_offset)
        )

        result = runner.run(seed=final_seed)

        # --------------------------------------------------------
        # Fitness definition
        # --------------------------------------------------------
        # By default: total accumulated reward.
        # Users may subclass Evaluator for custom fitness metrics.
        return float(result.total_reward)
