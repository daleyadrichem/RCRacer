"""
cmaes_trainer.py

Generic CMA-ES trainer for optimizing continuous genomes.

SIMULATION Layer
----------------
- Controller-agnostic (does not import agents/controllers)
- Environment-agnostic (does not import env/core)
- Deterministic given a fixed seed
- No shared mutable global state
- Parallelism is optional and handled via an injected mapping function

Notes
-----
This module wraps the external `cmaes` package's CMA optimizer.

The optimizer minimizes an objective, but in most racing tasks we maximize a
fitness. This trainer therefore expects `evaluate_fn` to return a fitness
(higher is better) and internally converts it to a loss (lower is better) for
CMA-ES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from cmaes import CMA
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
Candidate = FloatArray
Fitness = float
Loss = float


@dataclass(frozen=True)
class CMAESTrainerConfig:
    """
    Configuration for CMA-ES training.

    Parameters
    ----------
    genome_size : int
        Dimensionality of the genome vector.
    sigma : float
        Initial CMA-ES step-size.
    population_size : int
        Population size per generation.
    generations : int
        Number of generations to run.
    seed : int
        Random seed for CMA-ES (determinism).
    mean_init : ndarray | None
        Optional initial mean vector. If None, uses zeros.
    """

    genome_size: int
    sigma: float
    population_size: int
    generations: int
    seed: int
    mean_init: Optional[FloatArray] = None


@dataclass(frozen=True)
class GenerationStats:
    """
    Summary statistics for a single generation.

    Parameters
    ----------
    generation : int
        Generation index.
    best_fitness_gen : float
        Best fitness found in this generation.
    mean_fitness_gen : float
        Mean fitness in this generation.
    best_fitness_all : float
        Best fitness found across all generations so far.
    sigma : float
        Current CMA-ES sigma (step-size).
    """

    generation: int
    best_fitness_gen: float
    mean_fitness_gen: float
    best_fitness_all: float
    sigma: float


@dataclass(frozen=True)
class CMAESRunResult:
    """
    Result of a CMA-ES run.

    Parameters
    ----------
    best_genome : ndarray
        Best genome found (raw CMA space).
    best_fitness : float
        Fitness of best genome found.
    history : list[GenerationStats]
        Per-generation statistics.
    """

    best_genome: FloatArray
    best_fitness: float
    history: List[GenerationStats] = field(default_factory=list)


def _default_map(fn: Callable[[Candidate], Fitness], xs: Sequence[Candidate]) -> List[Fitness]:
    """
    Default sequential mapping function.

    Parameters
    ----------
    fn : Callable
        Function mapping a candidate to a fitness.
    xs : Sequence[ndarray]
        Candidates.

    Returns
    -------
    list[float]
        Fitness values.
    """
    return [float(fn(x)) for x in xs]


class CMAESTrainer:
    """
    Generic CMA-ES trainer.

    This class is deliberately controller-agnostic. It only deals with
    floating-point genomes and a user-supplied evaluation function.

    Examples
    --------
    >>> trainer = CMAESTrainer(CMAESTrainerConfig(...))
    >>> result = trainer.optimize(evaluate_fn=my_fitness_fn)
    >>> best_z = result.best_genome
    """

    def __init__(self, config: CMAESTrainerConfig) -> None:
        """
        Initialize the trainer.

        Parameters
        ----------
        config : CMAESTrainerConfig
            Immutable CMA-ES configuration.
        """
        self._cfg: CMAESTrainerConfig = config

        if self._cfg.genome_size <= 0:
            raise ValueError("genome_size must be positive.")
        if self._cfg.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        if self._cfg.population_size <= 1:
            raise ValueError("population_size must be > 1.")
        if self._cfg.generations <= 0:
            raise ValueError("generations must be positive.")

        mean_init: FloatArray
        if self._cfg.mean_init is None:
            mean_init = np.zeros((self._cfg.genome_size,), dtype=np.float64)
        else:
            mean_init = np.asarray(self._cfg.mean_init, dtype=np.float64)
            if mean_init.shape != (self._cfg.genome_size,):
                raise ValueError(
                    f"mean_init must have shape ({self._cfg.genome_size},), got {mean_init.shape}."
                )

        self._optimizer: CMA = CMA(
            mean=mean_init,
            sigma=float(self._cfg.sigma),
            population_size=int(self._cfg.population_size),
            seed=int(self._cfg.seed),
        )

    @property
    def config(self) -> CMAESTrainerConfig:
        """
        Return the trainer configuration.

        Returns
        -------
        CMAESTrainerConfig
        """
        return self._cfg

    @property
    def optimizer(self) -> CMA:
        """
        Return the underlying CMA optimizer instance.

        Returns
        -------
        cmaes.CMA
        """
        return self._optimizer

    def optimize(
        self,
        *,
        evaluate_fn: Callable[[Candidate], Fitness],
        map_fn: Optional[Callable[[Callable[[Candidate], Fitness], Sequence[Candidate]], List[Fitness]]] = None,
        on_generation: Optional[Callable[[GenerationStats, Candidate, Fitness], None]] = None,
        verbose: bool = True,
    ) -> CMAESRunResult:
        """
        Run CMA-ES optimization.

        Parameters
        ----------
        evaluate_fn : Callable
            Fitness function mapping a candidate genome vector (ndarray) to a
            scalar fitness (higher is better).
        map_fn : Callable | None
            Optional mapping function for evaluating a batch of candidates.
            Signature: map_fn(evaluate_fn, candidates) -> list[fitness].
            If None, uses a deterministic sequential map.
            Use this to inject parallel execution (e.g., ProcessPoolExecutor).
        on_generation : Callable | None
            Optional callback invoked after each generation:
            on_generation(stats, best_genome_so_far, best_fitness_so_far).
        verbose : bool
            If True, print per-generation summary.

        Returns
        -------
        CMAESRunResult
            Best genome, fitness, and history.

        Notes
        -----
        - CMA-ES *minimizes* loss. This trainer converts fitness to loss by
          using loss = -fitness.
        - For multiprocessing, `evaluate_fn` must be picklable.
        """
        mapper = map_fn if map_fn is not None else _default_map

        best_fitness: float = -np.inf
        best_genome: Optional[FloatArray] = None
        history: List[GenerationStats] = []

        for gen in range(int(self._cfg.generations)):
            # Ask for candidates
            candidates: List[FloatArray] = [
                np.asarray(self._optimizer.ask(), dtype=np.float64)
                for _ in range(int(self._optimizer.population_size))
            ]

            # Evaluate (fitness)
            fitnesses: List[float] = mapper(evaluate_fn, candidates)
            if len(fitnesses) != len(candidates):
                raise RuntimeError("map_fn returned a mismatched number of fitness values.")

            # Convert to (candidate, loss) pairs for CMA
            solutions: List[Tuple[FloatArray, Loss]] = []
            for z, fit in zip(candidates, fitnesses):
                fit_f = float(fit)
                solutions.append((z, -fit_f))

                if fit_f > best_fitness:
                    best_fitness = fit_f
                    best_genome = z.copy()

            self._optimizer.tell(solutions)

            best_fitness_gen = float(np.max(np.asarray(fitnesses, dtype=np.float64)))
            mean_fitness_gen = float(np.mean(np.asarray(fitnesses, dtype=np.float64)))

            # `cmaes.CMA` exposes sigma as `_sigma` (private) in the version you hit.
            sigma = float(getattr(self._optimizer, "_sigma", np.nan))

            stats = GenerationStats(
                generation=gen,
                best_fitness_gen=best_fitness_gen,
                mean_fitness_gen=mean_fitness_gen,
                best_fitness_all=float(best_fitness),
                sigma=sigma,
            )
            history.append(stats)

            if verbose:
                print(
                    f"Gen {gen:03d} | "
                    f"best(gen)={best_fitness_gen:10.3f} | "
                    f"mean(gen)={mean_fitness_gen:10.3f} | "
                    f"best(all)={best_fitness:10.3f} | "
                    f"sigma={sigma:10.6f} | "
                    f"genome={best_genome}"
                )

            if on_generation is not None and best_genome is not None:
                on_generation(stats, best_genome, float(best_fitness))

        if best_genome is None:
            raise RuntimeError("CMA-ES produced no best genome (unexpected).")

        return CMAESRunResult(
            best_genome=best_genome,
            best_fitness=float(best_fitness),
            history=history,
        )
