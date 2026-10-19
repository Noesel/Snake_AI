import random
import unittest
from unittest.mock import patch

from tests.support import Bait, make_snake


class BaitTests(unittest.TestCase):
    def test_food_stays_on_free_grid_cells_in_each_board(self):
        rng = random.Random(7)
        with patch.object(random, "randint", side_effect=rng.randint):
            for row in range(3):
                for column in range(8):
                    start_x, start_y = column * 230, 20 + row * 280
                    snake = make_snake(start_x, start_y)
                    bait = Bait(start_x, start_y, 220, 220, 10, 20)
                    for _ in range(20):
                        bait.generate_pos(snake)
                        self.assertNotIn((bait.x, bait.y), snake.body_parts)
                        self.assertGreater(bait.x, start_x)
                        self.assertLess(bait.x, start_x + 220)
                        self.assertGreater(bait.y, start_y + 20)
                        self.assertLess(bait.y, start_y + 240)
                        self.assertEqual((bait.x - start_x) % 10, 0)
                        self.assertEqual((bait.y - start_y - 20) % 10, 0)

    def test_food_retries_when_a_cell_is_occupied(self):
        snake = make_snake()
        bait = Bait(0, 20, 220, 220, 10, 20)
        with patch.object(random, "randint", side_effect=[10, 10, 1, 1]) as randint:
            bait.generate_pos(snake)
        self.assertEqual(randint.call_count, 4)
        self.assertEqual((bait.x, bait.y), (10, 50))
        for call in randint.call_args_list:
            self.assertTrue(all(isinstance(bound, int) for bound in call.args))

    def test_full_board_terminates_food_placement(self):
        snake = make_snake()
        snake.body_parts = [(x, y) for y in range(50, 260, 10) for x in range(10, 220, 10)]
        bait = Bait(0, 20, 220, 220, 10, 20, rng=random.Random(1))
        self.assertFalse(bait.generate_pos(snake))
        self.assertEqual((bait.x, bait.y), (None, None))

    def test_dense_board_fallback_finds_the_only_free_cell(self):
        snake = make_snake()
        snake.body_parts = [(x, y) for y in range(50, 260, 10) for x in range(10, 220, 10)
                            if (x, y) != (210, 250)]
        bait = Bait(0, 20, 220, 220, 10, 20, rng=random.Random(1))
        with patch.object(bait.rng, 'randint', return_value=1):
            self.assertTrue(bait.generate_pos(snake))
        self.assertEqual((bait.x, bait.y), (210, 250))
