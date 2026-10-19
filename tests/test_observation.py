import configparser
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from tests.support import make_game


class ObservationTests(unittest.TestCase):
    def make_scene(self, heading=0, start_x=0, start_y=20):
        game = make_game(start_x, start_y)
        game.snake.direction = heading
        game.bait = SimpleNamespace(x=game.snake.x_head_pos + 30,
                                    y=game.snake.y_head_pos - 40)
        return game

    def test_food_direction_rotates_with_the_snake(self):
        # Food is northeast of the head; expected signs are forward and right.
        expected = {0: (1, 1), 90: (1, -1), 180: (-1, -1), 270: (-1, 1)}
        for heading, signs in expected.items():
            with self.subTest(heading=heading):
                self.assertEqual(self.make_scene(heading).get_observation()[3:], signs)

    def test_food_ahead_behind_left_and_right_are_distinguishable(self):
        game = self.make_scene()
        x, y = game.snake.x_head_pos, game.snake.y_head_pos
        cases = [((x, y - 30), (1, 0)), ((x, y + 30), (-1, 0)),
                 ((x - 30, y), (0, -1)), ((x + 30, y), (0, 1))]
        for (game.bait.x, game.bait.y), expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(game.get_observation()[3:], expected)

    def test_food_distance_does_not_change_the_direction_signals(self):
        game = self.make_scene()
        before = game.get_observation()[3:]
        game.bait.x = game.snake.x_head_pos + 60
        game.bait.y = game.snake.y_head_pos - 80
        self.assertEqual(game.get_observation()[3:], before)

    def test_observation_is_identical_on_every_displayed_board(self):
        for heading in (0, 90, 180, 270):
            expected = self.make_scene(heading).get_observation()
            for row in range(3):
                for column in range(8):
                    with self.subTest(heading=heading, row=row, column=column):
                        game = self.make_scene(heading, column * 230, 20 + row * 280)
                        self.assertEqual(game.get_observation(), expected)

    def assert_danger_matches_movement(self, game):
        danger = game.get_observation()[:3]
        for index, action in enumerate((None, "turn_left", "turn_right")):
            probe = copy.deepcopy(game.snake)
            if action is not None:
                getattr(probe, action)()
            probe.move()
            self.assertEqual(danger[index], int(probe.collided),
                             f"Incorrect danger for {action or 'straight'}")

    def test_wall_danger_matches_actual_collisions_for_all_headings_and_boards(self):
        for row in range(3):
            for column in range(8):
                for x, y in ((10, 30), (210, 30), (10, 230), (210, 230), (100, 120)):
                    for heading in (0, 90, 180, 270):
                        with self.subTest(row=row, column=column, x=x, y=y, heading=heading):
                            game = self.make_scene(heading, column * 230, 20 + row * 280)
                            game.snake.x_head_pos = game.start_point_x + x
                            game.snake.y_head_pos = game.start_point_y + y
                            game.snake.body_parts = [(game.snake.x_head_pos, game.snake.y_head_pos)]
                            self.assert_danger_matches_movement(game)

    def test_body_danger_matches_actual_collisions_for_all_headings(self):
        # A valid body blocks east and south; north and west remain free.
        body = [(100, 140), (100, 150), (110, 150), (110, 140),
                (120, 140)]
        expected = {0: (0, 0, 1), 90: (1, 0, 1),
                    180: (1, 1, 0), 270: (0, 1, 0)}
        for heading in (0, 90, 180, 270):
            with self.subTest(heading=heading):
                game = self.make_scene(heading)
                game.snake.body_parts = list(body)
                self.assertEqual(game.get_observation()[:3], expected[heading])
                self.assert_danger_matches_movement(game)

    def test_vacating_tail_cell_is_safe(self):
        game = self.make_scene(90)
        game.snake.body_parts = [(100, 140), (100, 150), (110, 150), (110, 140)]
        self.assertEqual(game.get_observation()[0], 0)
        self.assert_danger_matches_movement(game)

    def test_observing_does_not_change_the_game_state(self):
        game = self.make_scene()
        before = copy.deepcopy((vars(game.snake), vars(game.bait)))
        game.get_observation()
        self.assertEqual((vars(game.snake), vars(game.bait)), before)

    def test_configuration_matches_the_five_bounded_inputs(self):
        config = configparser.ConfigParser()
        config.read(Path(__file__).resolve().parents[1] / "config-feedforward.txt")
        observation = self.make_scene().get_observation()
        self.assertEqual(len(observation), 5)
        self.assertEqual(config.getint("DefaultGenome", "num_inputs"), len(observation))
        self.assertTrue(all(value in (0, 1) for value in observation[:3]))
        self.assertTrue(all(value in (-1, 0, 1) for value in observation[3:]))
