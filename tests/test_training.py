from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.support import training


class TrainingLifecycleTests(unittest.TestCase):
    def run_generation(self, lifetimes, death_causes, food_on_first_tick=False, network_output=0):
        created = []
        genomes = [SimpleNamespace(index=i, fitness=None, observations=[]) for i in range(24)]
        testcase = self

        class FakeGame:
            def __init__(self, win, index, start_x, start_y):
                self.index = index
                self.ticks = 0
                self.STAT_FONT = Mock()
                self.snake = SimpleNamespace(energy=20, collided=False, x_head_pos=0,
                                            y_head_pos=0, turn_right=Mock(), turn_left=Mock())
                self.bait = SimpleNamespace(x=10, y=10)
                self.got_bait = False
                created.append(self)

            def run(self):
                if self.snake.collided or self.snake.energy <= 0:
                    raise AssertionError("A terminated game received an extra tick")
                if network_output > 0.5:
                    testcase.assertEqual(self.snake.turn_right.call_count, self.ticks + 1)
                elif network_output < -0.5:
                    testcase.assertEqual(self.snake.turn_left.call_count, self.ticks + 1)
                self.ticks += 1
                self.snake.energy -= 1
                self.got_bait = food_on_first_tick and self.ticks == 1
                if self.ticks == lifetimes[self.index]:
                    if death_causes[self.index] == "collision":
                        self.snake.collided = True
                    else:
                        self.snake.energy = 0

            def get_observation(self):
                # The ID is a sentinel to detect mismatched games and networks.
                return self.index, 0, 0, 1, -1

        def make_network(genome, config):
            def activate(observation):
                self.assertEqual(len(observation), 5)
                self.assertEqual(observation[0], genome.index,
                                 "Game and genome were paired incorrectly")
                genome.observations.append(observation)
                return [network_output]
            return SimpleNamespace(activate=activate)

        with patch.object(training, "Game", FakeGame), \
             patch.object(training, "GEN", 0), \
             patch.object(training.pygame.event, "get", return_value=[]), \
             patch.object(training.neat.nn.FeedForwardNetwork, "create", side_effect=make_network):
            training.main(list(enumerate(genomes)), None)

        self.assertEqual(len(created), 24)
        for game, genome in zip(created, genomes):
            self.assertEqual(game.ticks, lifetimes[genome.index])
            self.assertEqual(len(genome.observations), game.ticks)
            self.assertEqual(genome.fitness, 90 if food_on_first_tick else -10)

    def test_simultaneous_energy_deaths_are_removed_before_the_next_tick(self):
        self.run_generation([1] * 24, ["energy"] * 24)

    def test_simultaneous_collisions_are_removed_before_the_next_tick(self):
        self.run_generation([1] * 24, ["collision"] * 24)

    def test_mixed_lifetimes_keep_games_networks_and_fitness_aligned(self):
        self.run_generation([2 + i % 3 for i in range(24)],
                            ["collision" if i % 2 else "energy" for i in range(24)],
                            food_on_first_tick=True)

    def test_controller_acts_before_the_first_and_every_subsequent_move(self):
        for output in (-1, 1):
            with self.subTest(output=output):
                self.run_generation([2] * 24, ["energy"] * 24, network_output=output)
