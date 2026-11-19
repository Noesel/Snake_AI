"""Typed settings shared by all methods, loaded from TOML tables."""

import random
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EnvSettings:
    """How a method sees the game; must be identical for compared methods."""

    local_awareness: bool = False
    max_episode_steps: int = 10_000
    """External step limit (truncation), a safety net above the energy rule."""

    def __post_init__(self) -> None:
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Fixed evaluation games, separate from training seeds."""

    episodes: int = 20
    seed: int = 20_261_119
    max_steps: int = 10_000

    def __post_init__(self) -> None:
        if self.episodes < 1 or self.max_steps < 1:
            raise ValueError("episodes and max_steps must be positive")

    def game_seeds(self) -> tuple[int, ...]:
        """The same seeds for every method and snapshot."""
        rng = random.Random(self.seed)
        return tuple(rng.randrange(2**63) for _ in range(self.episodes))


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def from_table[T](cls: type[T], table: Mapping[str, Any]) -> T:
    """Build a settings dataclass from a TOML table, rejecting unknown keys."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    unknown = set(table) - {item.name for item in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {', '.join(sorted(unknown))}")
    return cls(**table)
