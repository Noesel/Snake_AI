"""Demo model: several methods play the same games side by side.

Each slot shows one method and can switch between the snapshots of one run.
All slots play the same game seed, taken from the fixed evaluation seeds, so
what is on screen matches the evaluation numbers. No pygame here; the window
and key handling live in the CLI.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from snake_ai.core import GameConfig, GameState, ObservationEncoder, SnakeGame
from snake_ai.policies import Policy
from snake_ai.runs import SnapshotMeta

logger = logging.getLogger(__name__)

type PolicyLoader = Callable[[Path], tuple[Policy, SnapshotMeta]]
type StartAt = Literal["first", "last", "best"]


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    path: Path
    meta: SnapshotMeta
    evaluation: dict[str, Any] | None
    """The evaluation record from ``metrics.jsonl``, if present."""

    @property
    def mean_score(self) -> float | None:
        return None if self.evaluation is None else self.evaluation["mean_score"]


def load_run(run: Path, game_config: GameConfig) -> list[SnapshotEntry]:
    """Compatible snapshots of one run, ordered by progress."""
    evaluations = _evaluations(run / "metrics.jsonl")
    entries = []
    for directory in sorted((run / "snapshots").glob("*/")):
        meta = SnapshotMeta.load(directory)
        try:
            meta.check_compatible(game_config)
        except ValueError as error:
            # All snapshots of a run share one layout, so one mismatch rules it out.
            logger.warning("skipping run %s: %s", run, error)
            return []
        entries.append(SnapshotEntry(directory, meta, evaluations.get(meta.progress)))
    return sorted(entries, key=lambda entry: entry.meta.progress)


def latest_run(root: Path, method: str, game_config: GameConfig) -> list[SnapshotEntry]:
    """Snapshots of the newest run of ``method`` that fits the game; may be empty."""
    # Run directory names start with a UTC timestamp, so name order is time order.
    for run in sorted((root / method).glob("*/"), reverse=True):
        entries = load_run(run, game_config)
        if entries:
            return entries
    return []


def start_index(entries: Sequence[SnapshotEntry], start: StartAt) -> int:
    if start == "first" or not entries:
        return 0
    if start == "last":
        return len(entries) - 1
    scored = [i for i, entry in enumerate(entries) if entry.mean_score is not None]
    if not scored:
        return len(entries) - 1
    # Ties go to the later snapshot.
    return max(scored, key=lambda i: (entries[i].mean_score, i))


def format_progress(meta: SnapshotMeta) -> str:
    if meta.unit == "generations":
        return f"gen {meta.progress}"
    if meta.progress >= 1_000_000:
        return f"{meta.progress / 1_000_000:.1f}M steps"
    if meta.progress >= 1_000:
        return f"{meta.progress / 1_000:.0f}k steps"
    return f"{meta.progress} steps"


class DemoSlot:
    """One method's board: a game plus the snapshot that currently plays it."""

    def __init__(
        self,
        method: str,
        entries: Sequence[SnapshotEntry],
        load: PolicyLoader,
        game_config: GameConfig,
        max_steps: int,
        index: int = 0,
    ) -> None:
        self.method = method
        self._entries = tuple(entries)
        self._load = load
        self._config = game_config
        self._max_steps = max_steps
        self._index = min(max(index, 0), max(len(self._entries) - 1, 0))
        self._game = SnakeGame(game_config)
        self._policies: dict[Path, tuple[Policy, ObservationEncoder]] = {}
        self._seed = 0
        self._game.reset(self._seed)

    @property
    def entry(self) -> SnapshotEntry | None:
        return self._entries[self._index] if self._entries else None

    @property
    def state(self) -> GameState:
        return self._game.state

    @property
    def done(self) -> bool:
        state = self._game.state
        return self.entry is None or state.done or state.steps >= self._max_steps

    def restart(self, seed: int) -> None:
        self._seed = seed
        self._game.reset(seed)

    def select(self, delta: int) -> None:
        """Move to an earlier or later snapshot and replay the current game."""
        index = min(max(self._index + delta, 0), max(len(self._entries) - 1, 0))
        if index != self._index:
            self._index = index
            self.restart(self._seed)

    def step(self) -> None:
        entry = self.entry
        if entry is None or self.done:
            return
        policy, encoder = self._policy(entry)
        self._game.step(policy.act(encoder.encode(self._game.state)))

    def title(self) -> str:
        entry = self.entry
        if entry is None:
            return self.method.upper()
        return f"{self.method.upper()}  {format_progress(entry.meta)}"

    def subtitle(self) -> str:
        entry = self.entry
        if entry is None:
            return "no compatible snapshot"
        position = f"[{self._index + 1}/{len(self._entries)}]"
        if entry.evaluation is None:
            return f"eval n/a  {position}"
        return (
            f"eval mean {entry.evaluation['mean_score']:.1f} "
            f"max {entry.evaluation['max_score']}  {position}"
        )

    def _policy(self, entry: SnapshotEntry) -> tuple[Policy, ObservationEncoder]:
        if entry.path not in self._policies:
            policy, meta = self._load(entry.path)
            encoder = ObservationEncoder(self._config, local_awareness=meta.local_awareness)
            self._policies[entry.path] = policy, encoder
        return self._policies[entry.path]


class Demo:
    """All slots on the same game; the game index walks the evaluation seeds."""

    def __init__(self, slots: Sequence[DemoSlot], seeds: Sequence[int], game: int = 0) -> None:
        if not slots or not seeds:
            raise ValueError("demo needs at least one slot and one seed")
        self.slots = tuple(slots)
        self._seeds = tuple(seeds)
        self._game = game % len(self._seeds)
        self.selected = 0
        self.restart()

    @property
    def all_done(self) -> bool:
        return all(slot.done for slot in self.slots)

    def game_label(self) -> str:
        return f"Game {self._game + 1}/{len(self._seeds)}"

    def restart(self) -> None:
        for slot in self.slots:
            slot.restart(self._seeds[self._game])

    def change_game(self, delta: int) -> None:
        self._game = (self._game + delta) % len(self._seeds)
        self.restart()

    def step(self) -> None:
        for slot in self.slots:
            slot.step()

    def select_slot(self, index: int) -> None:
        if 0 <= index < len(self.slots):
            self.selected = index

    def cycle_slot(self, delta: int) -> None:
        self.selected = (self.selected + delta) % len(self.slots)

    def change_snapshot(self, delta: int) -> None:
        self.slots[self.selected].select(delta)


def _evaluations(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    records = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {
        record["progress"]: record["evaluation"] for record in records if "evaluation" in record
    }
