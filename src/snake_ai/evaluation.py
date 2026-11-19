"""Play policies on the fixed evaluation seeds and summarize the results."""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from snake_ai.core import EndReason, GameConfig, ObservationEncoder, SnakeGame
from snake_ai.policies import Policy
from snake_ai.settings import EnvSettings, EvaluationSettings

TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int
    score: int
    steps: int
    end_reason: str
    """An ``EndReason`` value or ``"truncated"`` for the external step limit."""


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    episodes: tuple[EpisodeResult, ...]

    @property
    def mean_score(self) -> float:
        return sum(episode.score for episode in self.episodes) / len(self.episodes)

    @property
    def mean_steps(self) -> float:
        return sum(episode.steps for episode in self.episodes) / len(self.episodes)

    @property
    def win_rate(self) -> float:
        wins = sum(episode.end_reason == EndReason.BOARD_FULL.value for episode in self.episodes)
        return wins / len(self.episodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "mean_score": self.mean_score,
            "max_score": max(episode.score for episode in self.episodes),
            "mean_steps": self.mean_steps,
            "win_rate": self.win_rate,
            "end_reasons": dict(Counter(episode.end_reason for episode in self.episodes)),
        }


def play_episode(
    policy: Policy, game: SnakeGame, encoder: ObservationEncoder, seed: int, max_steps: int
) -> EpisodeResult:
    state = game.reset(seed)
    while not state.done and state.steps < max_steps:
        game.step(policy.act(encoder.encode(state)))
        state = game.state
    end_reason = state.end_reason.value if state.end_reason is not None else TRUNCATED
    return EpisodeResult(seed, state.score, state.steps, end_reason)


def evaluate(
    policy: Policy,
    game_config: GameConfig,
    env_settings: EnvSettings,
    settings: EvaluationSettings,
) -> EvaluationSummary:
    game = SnakeGame(game_config)
    encoder = ObservationEncoder(game_config, local_awareness=env_settings.local_awareness)
    max_steps = min(settings.max_steps, env_settings.max_episode_steps)
    return EvaluationSummary(
        tuple(
            play_episode(policy, game, encoder, seed, max_steps)
            for seed in settings.game_seeds()
        )
    )
