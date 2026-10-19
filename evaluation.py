"""Deterministic CPU evaluation; no pygame imports or display initialization."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import random
import signal
import time

from c_Game import Game


@dataclass(frozen=True)
class EpisodeResult:
    score: int
    steps: int

    @property
    def fitness(self):
        return self.score * 100 - 10


@dataclass(frozen=True)
class GenomeResult:
    episodes: tuple

    @property
    def fitness(self):
        return sum(episode.fitness for episode in self.episodes) / len(self.episodes)


def step_game(game, network):
    """Use the same action, movement, and energy rules in both render modes."""
    output = network.activate(game.get_observation())[0]
    if output > 0.5:
        game.snake.turn_right()
    elif output < -0.5:
        game.snake.turn_left()
    game.run()
    if game.got_bait:
        game.snake.energy += 20


def play_episode(network, seed, max_steps):
    game = Game(None, 0, 0, 0, rng=random.Random(seed))
    steps = 0
    while not game.done and steps < max_steps:
        step_game(game, network)
        steps += 1
    return EpisodeResult(game.score, steps)


def evaluate_genome(genome, config, seeds, max_steps):
    import neat

    network = neat.nn.FeedForwardNetwork.create(genome, config)
    return GenomeResult(tuple(play_episode(network, seed, max_steps) for seed in seeds))


_worker_context = None


def _initialize_worker(config, seeds, max_steps):
    global _worker_context
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_context = config, seeds, max_steps


def _evaluate_task(item):
    genome_id, genome = item
    config, seeds, max_steps = _worker_context
    return genome_id, evaluate_genome(genome, config, seeds, max_steps)


class PopulationEvaluator:
    """One genome per task, with a persistent pool and shared evaluation seeds."""

    def __init__(self, config, seeds, max_steps, workers=1, renderer=None):
        if not seeds or max_steps < 1 or workers < 1:
            raise ValueError("Evaluation needs seeds, positive max_steps, and positive workers")
        if renderer is not None and workers != 1:
            raise ValueError("Rendering runs in one process; use headless for multiple workers")
        self.config = config
        self.seeds = tuple(seeds)
        self.max_steps = max_steps
        self.workers = workers
        self.renderer = renderer
        self.pool = None
        self.results = {}
        self.seconds = 0

    def __enter__(self):
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=(self.config, self.seeds, self.max_steps),
            )
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
        if self.renderer is not None:
            self.renderer.close()

    def evaluate(self, genomes, config):
        started = time.perf_counter()
        items = list(genomes)
        if self.renderer is not None:
            results = self.renderer.evaluate(items, config, self.seeds, self.max_steps)
        elif self.workers == 1:
            results = {gid: evaluate_genome(genome, config, self.seeds, self.max_steps)
                       for gid, genome in items}
        else:
            if self.pool is None:
                raise RuntimeError("Use PopulationEvaluator as a context manager")
            results = dict(self.pool.map(_evaluate_task, items, chunksize=1))
        self.results = results
        self.seconds = time.perf_counter() - started
        for genome_id, genome in items:
            genome.fitness = results[genome_id].fitness
