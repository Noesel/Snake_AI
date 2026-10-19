"""Graphics-free fixtures shared by the game logic tests."""

from c_Bait import Bait
from c_Game import Game
from c_Snake import Snake


def make_snake(start_x=0, start_y=20):
    return Snake(start_x, start_y, start_x + 220, start_y + 220, 10, 20)


def make_game(start_x=0, start_y=20):
    return Game(None, 0, start_x, start_y)
