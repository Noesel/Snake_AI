import unittest

from tests.support import make_snake


class SnakeDirectionTests(unittest.TestCase):
    def test_turns_produce_the_expected_physical_movement(self):
        movement = {0: (0, -10), 90: (10, 0), 180: (0, 10), 270: (-10, 0)}
        clockwise = [(0, 90), (90, 180), (180, 270), (270, 0)]
        counterclockwise = [(0, 270), (90, 0), (180, 90), (270, 180)]
        for method, cases in (("turn_right", clockwise), ("turn_left", counterclockwise)):
            for initial, expected in cases:
                with self.subTest(method=method, initial=initial):
                    snake = make_snake()
                    snake.direction = initial
                    snake.body_parts = [snake.body_parts[0]]
                    before = snake.x_head_pos, snake.y_head_pos
                    getattr(snake, method)()
                    self.assertEqual(snake.direction, expected)
                    snake.move()
                    dx, dy = movement[expected]
                    self.assertEqual((snake.x_head_pos, snake.y_head_pos),
                                     (before[0] + dx, before[1] + dy))

    def test_repeated_full_rotations_return_to_the_starting_heading(self):
        for method in ("turn_right", "turn_left"):
            snake = make_snake()
            for _ in range(100):
                for _ in range(4):
                    getattr(snake, method)()
                    self.assertIn(snake.direction, (0, 90, 180, 270))
                self.assertEqual(snake.direction, 180)
