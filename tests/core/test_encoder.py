"""Observation layout; models silently break if it shifts."""

import numpy as np
import pytest

from snake_ai.core import Direction, GameConfig, GameState, ObservationEncoder

CONFIG = GameConfig()


def state(direction: Direction = Direction.UP, energy: int = 50) -> GameState:
    body = ((5, 5), (5, 6), (5, 7))
    return GameState(10, 10, body, (9, 0), direction, energy, 0, 0, None)


def test_layout_of_version_1() -> None:
    encoder = ObservationEncoder(CONFIG)
    observation = encoder.encode(state(Direction.RIGHT))

    assert encoder.size == observation.size == 107
    assert observation.dtype == np.float32
    grid = observation[:100].reshape(10, 10)
    assert grid[5, 5] == np.float32(3 / 100)  # head
    assert grid[7, 5] == np.float32(1 / 100)  # tail
    assert np.count_nonzero(grid) == 3
    # Food (9, 0) from head (5, 5) heading right: 4 ahead, 5 to the left.
    np.testing.assert_allclose(observation[100:102], [(4 / 10 + 1) / 2, (-5 / 10 + 1) / 2])
    np.testing.assert_array_equal(observation[102:106], [0, 1, 0, 0])  # heading right
    assert observation[106] == np.float32(0.5)


def test_local_awareness_flags_danger_ahead_left_right() -> None:
    encoder = ObservationEncoder(CONFIG, local_awareness=True)
    near_wall = GameState(10, 10, ((0, 0), (1, 0), (2, 0)), (9, 9), Direction.LEFT, 50, 0, 0, None)
    observation = encoder.encode(near_wall)
    assert encoder.size == 110
    # Heading left at the top-left corner: wall ahead, wall to the right
    # (up), free to the left (down).
    np.testing.assert_array_equal(observation[107:], [1, 0, 1])


def test_food_offset_follows_heading() -> None:
    encoder = ObservationEncoder(CONFIG)

    # Food directly above the head (5, 5): ahead when heading up, right when
    # heading left, behind when heading down.
    def offset(direction: Direction) -> list[float]:
        food_above = GameState(10, 10, ((5, 5), (5, 6)), (5, 2), direction, 50, 0, 0, None)
        return encoder.encode(food_above)[100:102].tolist()

    assert offset(Direction.UP) == pytest.approx([0.65, 0.5])
    assert offset(Direction.LEFT) == pytest.approx([0.5, 0.65])
    assert offset(Direction.DOWN) == pytest.approx([0.35, 0.5])
