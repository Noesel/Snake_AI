"""Rule edge cases from the decision log; easy to break, hard to see in training."""

from pathlib import Path

import pytest

from snake_ai.core import Action, Direction, EndReason, GameConfig, GameState, SnakeGame

CONFIG = GameConfig()


def scenario(
    body: list[tuple[int, int]],
    direction: Direction,
    food: tuple[int, int],
    *,
    energy: int = 100,
    config: GameConfig = CONFIG,
) -> SnakeGame:
    game = SnakeGame(config)
    state = GameState(config.width, config.height, tuple(body), food, direction, energy, 0, 0, None)
    game.reset_to(state, seed=0)
    return game


def test_reset_places_snake_in_centre_heading_up() -> None:
    state = SnakeGame(CONFIG).reset(seed=1)
    assert state.body == ((5, 5), (5, 6), (5, 7))
    assert state.direction is Direction.UP
    assert state.energy == 100
    assert state.food not in state.body


def test_same_seed_and_actions_reproduce_episode() -> None:
    actions = [Action.LEFT, Action.STRAIGHT, Action.RIGHT, Action.RIGHT] * 30

    def run() -> list[GameState]:
        game = SnakeGame(CONFIG)
        states = [game.reset(seed=7)]
        for action in actions:
            if game.step(action).terminated:
                break
            states.append(game.state)
        return states

    assert run() == run()


def test_relative_turns() -> None:
    assert Direction.UP.turned(Action.LEFT) is Direction.LEFT
    assert Direction.LEFT.turned(Action.LEFT) is Direction.DOWN
    assert Direction.UP.turned(Action.RIGHT) is Direction.RIGHT
    assert Direction.UP.turned(Action.STRAIGHT) is Direction.UP


def test_wall_collision_ends_episode() -> None:
    game = scenario([(5, 0), (5, 1), (5, 2)], Direction.UP, food=(0, 9))
    result = game.step(Action.STRAIGHT)
    assert result.end_reason is EndReason.WALL
    assert result.reward == -10


def test_self_collision_ends_episode() -> None:
    # Head at (2, 2) heading left; turning right runs into (2, 1).
    body = [(2, 2), (3, 2), (3, 1), (2, 1), (1, 1)]
    game = scenario(body, Direction.LEFT, food=(9, 9))
    result = game.step(Action.RIGHT)
    assert result.end_reason is EndReason.SELF
    assert result.reward == -10


def test_moving_into_vacating_tail_cell_is_allowed() -> None:
    # Square loop: the head moves onto the cell the tail leaves this step.
    body = [(2, 2), (3, 2), (3, 1), (2, 1)]
    game = scenario(body, Direction.LEFT, food=(9, 9))
    result = game.step(Action.RIGHT)
    assert result.end_reason is None
    assert game.state.body == ((2, 1), (2, 2), (3, 2), (3, 1))


def test_food_grows_snake_and_resets_energy() -> None:
    game = scenario([(5, 5), (5, 6), (5, 7)], Direction.UP, food=(5, 4), energy=40)
    result = game.step(Action.STRAIGHT)
    state = game.state
    assert result.ate_food
    assert result.reward == 10
    assert state.body == ((5, 4), (5, 5), (5, 6), (5, 7))
    assert state.energy == 100
    assert state.score == 1
    assert state.food not in state.body


def test_energy_depletion_ends_episode() -> None:
    game = scenario([(5, 5), (5, 6), (5, 7)], Direction.UP, food=(0, 9), energy=1)
    result = game.step(Action.STRAIGHT)
    assert result.end_reason is EndReason.ENERGY
    assert result.reward == -5


def test_food_on_last_energy_step_wins_over_depletion() -> None:
    game = scenario([(5, 5), (5, 6), (5, 7)], Direction.UP, food=(5, 4), energy=1)
    result = game.step(Action.STRAIGHT)
    assert result.end_reason is None
    assert game.state.energy == 100


def test_start_energy_allows_exactly_max_energy_moves() -> None:
    config = GameConfig(max_energy=4)
    game = SnakeGame(config)
    game.reset(seed=3)
    results = [game.step(action) for action in (Action.LEFT, Action.RIGHT) * 2]
    assert [result.end_reason for result in results] == [None, None, None, EndReason.ENERGY]


def test_filling_the_board_ends_with_bonus() -> None:
    config = GameConfig(width=2, height=2, start_length=1)
    game = scenario([(0, 0), (0, 1), (1, 1)], Direction.UP, food=(1, 0), config=config)
    result = game.step(Action.RIGHT)
    assert result.end_reason is EndReason.BOARD_FULL
    assert result.reward == 10 + 100
    assert game.state.food is None


def test_step_after_end_is_rejected() -> None:
    game = scenario([(5, 0), (5, 1), (5, 2)], Direction.UP, food=(0, 9))
    game.step(Action.STRAIGHT)
    with pytest.raises(RuntimeError):
        game.step(Action.STRAIGHT)


def test_shipped_config_matches_defaults() -> None:
    path = Path(__file__).parents[2] / "configs" / "game.toml"
    assert GameConfig.load(path) == GameConfig()


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="widht"):
        GameConfig.from_dict({"widht": 10})
