"""Gymnasium adapter for the RL trainers."""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from snake_ai.core import Action, GameConfig, Observation, ObservationEncoder, SnakeGame


class SnakeEnv(gym.Env[Observation, np.int64]):
    """Snake as a Gymnasium environment with ``Discrete(3)`` relative actions.

    ``terminated`` means the game rules ended the episode; ``truncated`` means
    the optional external step limit cut it off. ``info`` carries ``score``,
    ``steps`` and ``end_reason`` (``None`` while running or when truncated).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        config: GameConfig,
        *,
        local_awareness: bool = False,
        max_episode_steps: int | None = None,
    ) -> None:
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive or None")
        self.game = SnakeGame(config)
        self.encoder = ObservationEncoder(config, local_awareness=local_awareness)
        self.max_episode_steps = max_episode_steps
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.encoder.size,), np.float32)
        self.action_space = gym.spaces.Discrete(len(Action))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        # Derive the game seed from Gymnasium's generator so a seeded env
        # yields a reproducible sequence of episodes.
        game_seed = int(self.np_random.integers(2**63))
        state = self.game.reset(game_seed)
        return self.encoder.encode(state), self._info(game_seed=game_seed)

    def step(self, action: np.int64 | int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        result = self.game.step(Action(int(action)))
        state = self.game.state
        truncated = (
            not result.terminated
            and self.max_episode_steps is not None
            and state.steps >= self.max_episode_steps
        )
        return self.encoder.encode(state), result.reward, result.terminated, truncated, self._info()

    def _info(self, **extra: Any) -> dict[str, Any]:
        state = self.game.state
        end_reason = state.end_reason.value if state.end_reason is not None else None
        return {"score": state.score, "steps": state.steps, "end_reason": end_reason, **extra}
