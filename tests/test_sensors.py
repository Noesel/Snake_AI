import copy
import unittest

from tests.support import make_game


class BorderSensorTests(unittest.TestCase):
    def test_readings_match_safe_travel_until_actual_wall_collision(self):
        # Check all headings, every displayed board, and asymmetric/edge positions.
        # Sensor distance means travel to the last safe cell: zero beside a wall.
        for row in range(3):
            for column in range(8):
                for x, y in ((10, 30), (30, 40), (100, 120), (210, 230)):
                    for heading in (0, 90, 180, 270):
                        with self.subTest(row=row, column=column, x=x, y=y, heading=heading):
                            game = make_game(column * 230, 20 + row * 280)
                            game.snake.x_head_pos = game.start_point_x + x
                            game.snake.y_head_pos = game.start_point_y + y
                            game.snake.body_parts = [(game.snake.x_head_pos, game.snake.y_head_pos)]
                            game.snake.direction = heading
                            # Explicit forward/left/right headings for an independent oracle.
                            relative = {0: (0, 270, 90), 90: (90, 0, 180),
                                        180: (180, 90, 270), 270: (270, 180, 0)}
                            expected = []
                            for direction in relative[heading]:
                                probe = copy.deepcopy(game.snake)
                                probe.direction = direction
                                safe_steps = 0
                                for _ in range(23):
                                    probe.move()
                                    if probe.collided:
                                        break
                                    safe_steps += 1
                                self.assertTrue(probe.collided)
                                expected.append(safe_steps * game.move_distance)
                            self.assertEqual(game.calc_border_distance(), tuple(expected))

    def test_terminal_positions_do_not_report_negative_clearance(self):
        for heading, position in ((0, (100, 40)), (90, (220, 140)),
                                  (180, (100, 260)), (270, (0, 140))):
            with self.subTest(heading=heading):
                game = make_game()
                game.snake.direction = heading
                game.snake.x_head_pos, game.snake.y_head_pos = position
                distances = game.calc_border_distance()
                self.assertEqual(distances[0], 0)
                self.assertTrue(all(distance >= 0 for distance in distances))
