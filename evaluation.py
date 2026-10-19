"""Python reference rollouts used to verify the JAX simulation rules."""

from dataclasses import dataclass
import random

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
