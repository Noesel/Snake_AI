import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evaluation import EpisodeResult, GenomeResult, play_episode, step_game
from tests.support import make_game


class TrainingTests(unittest.TestCase):
    def test_controller_acts_before_movement(self):
        for output, expected in ((-1, (110, 140)), (0, (100, 150)), (1, (90, 140))):
            with self.subTest(output=output):
                game = make_game()
                network = SimpleNamespace(activate=lambda observation: [output])
                step_game(game, network)
                self.assertEqual((game.snake.x_head_pos, game.snake.y_head_pos), expected)

    def test_food_increases_energy_and_grows_before_next_food_is_placed(self):
        game = make_game()
        game.bait.x, game.bait.y = 100, 150
        def place_food(snake):
            self.assertEqual(len(snake.body_parts), 4)
            return True
        with patch.object(game.bait, 'generate_pos', side_effect=place_food):
            step_game(game, SimpleNamespace(activate=lambda observation: [0]))
        self.assertEqual(game.score, 1)
        self.assertEqual(game.snake.energy, 39)
        self.assertFalse(game.done)

    def test_terminated_game_does_not_receive_an_extra_move(self):
        for cause in ('collision', 'energy', 'completed'):
            with self.subTest(cause=cause):
                game = make_game()
                if cause == 'collision':
                    game.snake.collided = True
                elif cause == 'energy':
                    game.snake.energy = 0
                else:
                    game.completed = True
                before = (list(game.snake.body_parts), game.snake.energy, game.score)
                game.run()
                self.assertEqual((game.snake.body_parts, game.snake.energy, game.score), before)
                self.assertFalse(game.got_bait)

    def test_episode_is_deterministic_and_does_not_change_evolution_rng(self):
        network = SimpleNamespace(activate=lambda observation: [1])
        state = random.getstate()
        first = play_episode(network, 73, 100)
        second = play_episode(network, 73, 100)
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), state)

    def test_episode_step_limit_is_enforced(self):
        result = play_episode(SimpleNamespace(activate=lambda observation: [1]), 73, 1)
        self.assertEqual(result.steps, 1)

    def test_fitness_is_the_mean_over_episodes(self):
        result = GenomeResult((EpisodeResult(0, 20), EpisodeResult(2, 30)))
        self.assertEqual(result.fitness, 90)

    def test_headless_game_runs_without_site_packages_or_pygame(self):
        code = ('import sys; from types import SimpleNamespace; '
                'from evaluation import play_episode; '
                'play_episode(SimpleNamespace(activate=lambda observation: [0]), 7, 100); '
                'assert "pygame" not in sys.modules; assert "neat" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-S', '-c', code],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
