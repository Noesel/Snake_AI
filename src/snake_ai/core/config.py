"""Typed game configuration, loadable from TOML."""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Self


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Rewards per event; see the decision log for the agreed start values."""

    food: float = 10.0
    collision: float = -10.0
    energy_depleted: float = -5.0
    step: float = 0.0
    board_full_bonus: float = 100.0
    """Paid on top of ``food`` for the move that fills the board."""


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Rules shared by every training method."""

    width: int = 10
    height: int = 10
    start_length: int = 3
    max_energy: int = 100
    """Moves available per food search; eating resets energy to this value."""
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("board must be at least 2 x 2")
        if self.start_length < 1:
            raise ValueError("start_length must be positive")
        # The body starts below the head, which sits in the middle row.
        if self.start_length > self.height - self.height // 2:
            raise ValueError("start_length does not fit below the middle row")
        if self.max_energy < 1:
            raise ValueError("max_energy must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        """Build from a plain mapping, rejecting unknown keys."""
        values = dict(data)
        rewards = values.pop("rewards", {})
        if not isinstance(rewards, dict):
            raise ValueError("'rewards' must be a table")
        _reject_unknown(cls, values)
        _reject_unknown(RewardConfig, rewards)
        return cls(**values, rewards=RewardConfig(**rewards))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load the ``[game]`` table of a TOML file."""
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        return cls.from_dict(data.get("game", {}))


def _reject_unknown(target: type, values: dict[str, object]) -> None:
    unknown = set(values) - {item.name for item in fields(target)}
    if unknown:
        raise ValueError(f"unknown {target.__name__} keys: {', '.join(sorted(unknown))}")
