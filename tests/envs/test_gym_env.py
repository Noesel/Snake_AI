"""Gymnasium contract; SB3 relies on it."""

import pytest
from gymnasium.utils.env_checker import check_env

from snake_ai.core import GameConfig
from snake_ai.envs import SnakeEnv


def test_passes_gymnasium_env_checker() -> None:
    check_env(SnakeEnv(GameConfig()), skip_render_check=True)


def test_step_limit_truncates_without_terminating() -> None:
    env = SnakeEnv(GameConfig(), max_episode_steps=2)
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(1)
    assert not (terminated or truncated)
    _, _, terminated, truncated, info = env.step(1)
    assert truncated and not terminated
    assert info["end_reason"] is None


def test_seeded_resets_are_reproducible() -> None:
    first, second = SnakeEnv(GameConfig()), SnakeEnv(GameConfig())
    seeds_a = [first.reset(seed=5)[1]["game_seed"], first.reset()[1]["game_seed"]]
    seeds_b = [second.reset(seed=5)[1]["game_seed"], second.reset()[1]["game_seed"]]
    assert seeds_a == seeds_b


def test_rejects_invalid_step_limit() -> None:
    with pytest.raises(ValueError):
        SnakeEnv(GameConfig(), max_episode_steps=0)
