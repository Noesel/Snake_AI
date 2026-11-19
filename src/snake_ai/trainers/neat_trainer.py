"""NEAT through neat-python on the shared core and observation.

Fitness is the mean sum of episode rewards over a few training games, i.e. the
same reward signal as the RL methods but undiscounted. All genomes of one
generation play the same games; every generation draws new games, which keeps
evolution from overfitting a fixed seed set.
"""

import logging
import pickle
import random
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import neat
import numpy as np

from snake_ai.core import Action, GameConfig, Observation, ObservationEncoder, SnakeGame
from snake_ai.runs import RunDirectory, SnapshotMeta, SnapshotRecorder
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml

METHOD = "neat"
UNIT = "generations"
GENOME_FILE = "genome.pkl"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NeatSettings:
    neat_config: str
    generations: int
    episodes: int = 5
    seed: int = 0
    snapshot_every: int = 25
    workers: int = 24

    def __post_init__(self) -> None:
        for name in ("generations", "episodes", "snapshot_every", "workers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def load(cls, path: Path) -> tuple[Self, EnvSettings]:
        data = load_toml(path)
        return from_table(cls, data["train"]), from_table(EnvSettings, data.get("env", {}))


class NeatPolicy:
    """Feed-forward network of one genome; the largest output is the action."""

    def __init__(self, genome: neat.DefaultGenome, config: neat.Config) -> None:
        self._network = neat.nn.FeedForwardNetwork.create(genome, config)

    def act(self, observation: Observation) -> Action:
        return Action(int(np.argmax(self._network.activate(observation.tolist()))))


@dataclass(frozen=True, slots=True)
class _GenomeResult:
    fitness: float
    scores: tuple[int, ...]
    steps: int


def play_genome(
    genome: neat.DefaultGenome,
    neat_config: neat.Config,
    game_config: GameConfig,
    env_settings: EnvSettings,
    seeds: tuple[int, ...],
) -> _GenomeResult:
    """Play one game per seed and return the mean undiscounted reward sum."""
    policy = NeatPolicy(genome, neat_config)
    game = SnakeGame(game_config)
    encoder = ObservationEncoder(game_config, local_awareness=env_settings.local_awareness)
    totals, scores, steps = [], [], 0
    for seed in seeds:
        state = game.reset(seed)
        total = 0.0
        while not state.done and state.steps < env_settings.max_episode_steps:
            total += game.step(policy.act(encoder.encode(state))).reward
            state = game.state
        totals.append(total)
        scores.append(state.score)
        steps += state.steps
    return _GenomeResult(sum(totals) / len(totals), tuple(scores), steps)


_worker_context: tuple[neat.Config, GameConfig, EnvSettings] | None = None


def _init_worker(neat_config: neat.Config, game_config: GameConfig, env: EnvSettings) -> None:
    global _worker_context
    _worker_context = (neat_config, game_config, env)


def _play_task(task: tuple[int, neat.DefaultGenome, tuple[int, ...]]) -> tuple[int, _GenomeResult]:
    if _worker_context is None:
        raise RuntimeError("worker not initialized")
    genome_id, genome, seeds = task
    return genome_id, play_genome(genome, *_worker_context, seeds)


class NeatTrainer:
    def __init__(
        self,
        settings: NeatSettings,
        env_settings: EnvSettings,
        game_config: GameConfig,
        evaluation: EvaluationSettings,
        run: RunDirectory,
    ) -> None:
        self._settings = settings
        self._env_settings = env_settings
        self._game_config = game_config
        self._recorder = SnapshotRecorder(
            run,
            method=METHOD,
            unit=UNIT,
            seed=settings.seed,
            game_config=game_config,
            env_settings=env_settings,
            evaluation=evaluation,
        )
        self._neat_config = load_neat_config(Path(settings.neat_config))
        expected = ObservationEncoder(
            game_config, local_awareness=env_settings.local_awareness
        ).size
        genome_config = self._neat_config.genome_config
        if genome_config.num_inputs != expected or genome_config.num_outputs != len(Action):
            raise ValueError(
                f"{settings.neat_config}: need num_inputs = {expected} and "
                f"num_outputs = {len(Action)}, found {genome_config.num_inputs} and "
                f"{genome_config.num_outputs}"
            )
        self._seed_stream = random.Random(settings.seed)
        self._pool: ProcessPoolExecutor | None = None
        self._last_results: dict[int, _GenomeResult] = {}

    def train(self) -> None:
        settings = self._settings
        self._recorder.write_run_files(
            {**asdict(settings), "neat_ini": Path(settings.neat_config).read_text()}
        )
        population = neat.Population(self._neat_config, seed=settings.seed)
        reporter = _SnapshotReporter(self.snapshot, population, settings)
        population.add_reporter(reporter)
        with ProcessPoolExecutor(
            max_workers=settings.workers,
            initializer=_init_worker,
            initargs=(self._neat_config, self._game_config, self._env_settings),
        ) as pool:
            self._pool = pool
            population.run(self._evaluate_generation, settings.generations)

    def _evaluate_generation(
        self, genomes: list[tuple[int, neat.DefaultGenome]], config: neat.Config
    ) -> None:
        seeds = tuple(self._seed_stream.randrange(2**63) for _ in range(self._settings.episodes))
        tasks = [(genome_id, genome, seeds) for genome_id, genome in genomes]
        if self._pool is None:
            raise RuntimeError("evaluation runs only inside train()")
        chunk = max(1, len(tasks) // (self._settings.workers * 4))
        results = dict(self._pool.map(_play_task, tasks, chunksize=chunk))
        for genome_id, genome in genomes:
            genome.fitness = results[genome_id].fitness
        self._last_results = results

    def snapshot(self, generation: int, best: neat.DefaultGenome, species: int) -> None:
        results = self._last_results.values()
        scores = [score for result in results for score in result.scores]
        config = self._neat_config

        def save(directory: Path) -> None:
            with (directory / GENOME_FILE).open("wb") as handle:
                pickle.dump({"genome": best, "config": config}, handle)

        self._recorder.record(
            generation,
            NeatPolicy(best, config),
            save,
            best_fitness=best.fitness,
            mean_fitness=float(np.mean([result.fitness for result in results])),
            train_mean_score=float(np.mean(scores)),
            train_max_score=float(max(scores)),
            env_steps=float(sum(result.steps for result in results)),
            species=float(species),
        )


class _SnapshotReporter(neat.reporting.BaseReporter):
    """Snapshot the generation's best genome at generation 0, every N and last."""

    def __init__(
        self,
        snapshot: Callable[[int, neat.DefaultGenome, int], None],
        population: neat.Population,
        settings: NeatSettings,
    ) -> None:
        self._snapshot = snapshot
        self._population = population
        self._settings = settings

    def post_evaluate(
        self, config: neat.Config, population: Any, species: Any, best_genome: Any
    ) -> None:
        generation = self._population.generation
        last = generation == self._settings.generations - 1
        if generation % self._settings.snapshot_every == 0 or last:
            self._snapshot(generation, best_genome, len(species.species))
        else:
            logger.debug("generation %d best fitness %.2f", generation, best_genome.fitness)


def load_neat_config(path: Path) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(path),
    )


def load_policy(directory: Path, game_config: GameConfig) -> tuple[NeatPolicy, SnapshotMeta]:
    meta = SnapshotMeta.load(directory)
    meta.check_compatible(game_config)
    # Only load snapshot files this project wrote; pickle executes code.
    with (directory / GENOME_FILE).open("rb") as handle:
        payload = pickle.load(handle)
    return NeatPolicy(payload["genome"], payload["config"]), meta
