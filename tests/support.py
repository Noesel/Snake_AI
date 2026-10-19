"""Load production logic without requiring a display or installed AI libraries."""

import sys
from unittest.mock import MagicMock, patch


with patch.dict(sys.modules, {"pygame": MagicMock(), "neat": MagicMock()}):
    from c_Bait import Bait
    from c_Game import Game
    from c_Snake import Snake
    import Snake_Game as training


def make_snake(start_x=0, start_y=20):
    return Snake(start_x, start_y, start_x + 220, start_y + 220, 10, 20)


def make_game(start_x=0, start_y=20):
    game = Game.__new__(Game)
    game.start_point_x = start_x
    game.start_point_y = start_y
    game.end_point_x = start_x + 220
    game.end_point_y = start_y + 220
    game.offset = 20
    game.move_distance = 10
    game.snake = make_snake(start_x, start_y)
    return game
