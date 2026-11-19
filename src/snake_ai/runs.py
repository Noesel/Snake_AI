"""Run directories: effective configuration, versions, metrics and snapshots."""

import json
import logging
import platform
import secrets
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Self

from snake_ai.core import SIMULATION_VERSION, GameConfig, ObservationEncoder
from snake_ai.evaluation import EvaluationSummary, evaluate
from snake_ai.policies import Policy
from snake_ai.settings import EnvSettings, EvaluationSettings

logger = logging.getLogger(__name__)

_PACKAGES = ("numpy", "gymnasium", "torch", "stable-baselines3", "neat-python")


class RunDirectory:
    """``runs/<method>/<UTC timestamp>_<id>/`` with JSON artifacts."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, root: Path, method: str) -> Self:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        path = root / method / f"{stamp}_{secrets.token_hex(2)}"
        path.mkdir(parents=True)
        return cls(path)

    @property
    def name(self) -> str:
        return self.path.name

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.path / name).write_text(json.dumps(payload, indent=2) + "\n")

    def append_metrics(self, record: dict[str, Any]) -> None:
        with (self.path / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    def snapshot_dir(self, progress: int, unit: str) -> Path:
        path = self.path / "snapshots" / f"{unit}_{progress:012d}"
        path.mkdir(parents=True, exist_ok=False)
        return path


@dataclass(frozen=True, slots=True)
class SnapshotMeta:
    """Everything needed to check that a snapshot fits the current game."""

    method: str
    run: str
    progress: int
    unit: str
    """``env_steps`` for RL, ``generations`` for NEAT; not directly comparable."""
    seed: int
    board_width: int
    board_height: int
    local_awareness: bool
    observation_version: int
    simulation_version: int
    created_at: str

    @classmethod
    def describe(
        cls,
        *,
        method: str,
        run: str,
        progress: int,
        unit: str,
        seed: int,
        game_config: GameConfig,
        local_awareness: bool,
    ) -> Self:
        return cls(
            method=method,
            run=run,
            progress=progress,
            unit=unit,
            seed=seed,
            board_width=game_config.width,
            board_height=game_config.height,
            local_awareness=local_awareness,
            observation_version=ObservationEncoder.VERSION,
            simulation_version=SIMULATION_VERSION,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def save(self, directory: Path) -> None:
        (directory / "meta.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> Self:
        return cls(**json.loads((directory / "meta.json").read_text()))

    def check_compatible(self, game_config: GameConfig) -> None:
        """Raise if the snapshot was trained on a different input layout."""
        problems = []
        if (self.board_width, self.board_height) != (game_config.width, game_config.height):
            problems.append(
                f"board {self.board_width}x{self.board_height} vs "
                f"{game_config.width}x{game_config.height}"
            )
        if self.observation_version != ObservationEncoder.VERSION:
            problems.append(
                f"observation v{self.observation_version} vs v{ObservationEncoder.VERSION}"
            )
        if problems:
            raise ValueError("incompatible snapshot: " + "; ".join(problems))


def environment_versions() -> dict[str, Any]:
    """Python, library and simulation versions plus the git commit, if known."""
    packages = {}
    for name in _PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "simulation_version": SIMULATION_VERSION,
        "observation_version": ObservationEncoder.VERSION,
        "git_commit": _git("rev-parse", "HEAD"),
        # Uncommitted changes make the commit alone insufficient to reproduce.
        "git_dirty": _git_dirty(),
    }


def _git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=True, timeout=5
        )
    except OSError, subprocess.SubprocessError:
        return None
    return result.stdout.strip()


def _git_dirty() -> bool | None:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return None if status is None else bool(status)


class SnapshotRecorder:
    """Shared snapshot routine: files, metadata, evaluation, metrics, log line.

    Every trainer saves its own model format through ``save``; everything else
    is identical across methods so their metrics stay comparable.
    """

    def __init__(
        self,
        run: RunDirectory,
        *,
        method: str,
        unit: str,
        seed: int,
        game_config: GameConfig,
        env_settings: EnvSettings,
        evaluation: EvaluationSettings,
    ) -> None:
        self._run = run
        self._method = method
        self._unit = unit
        self._seed = seed
        self._game_config = game_config
        self._env_settings = env_settings
        self._evaluation = evaluation
        self._started = time.perf_counter()

    def write_run_files(self, train_settings: dict[str, Any]) -> None:
        """Store the effective configuration and the software versions."""
        self._run.write_json(
            "config.json",
            {
                "method": self._method,
                "game": asdict(self._game_config),
                "env": asdict(self._env_settings),
                "train": train_settings,
                "evaluation": asdict(self._evaluation),
            },
        )
        self._run.write_json("versions.json", environment_versions())

    def record(
        self,
        progress: int,
        policy: Policy,
        save: Callable[[Path], None],
        **train_metrics: float | None,
    ) -> EvaluationSummary:
        directory = self._run.snapshot_dir(progress, self._unit)
        save(directory)
        SnapshotMeta.describe(
            method=self._method,
            run=self._run.name,
            progress=progress,
            unit=self._unit,
            seed=self._seed,
            game_config=self._game_config,
            local_awareness=self._env_settings.local_awareness,
        ).save(directory)
        summary = evaluate(policy, self._game_config, self._env_settings, self._evaluation)
        evaluation = summary.to_dict()
        self._run.append_metrics(
            {
                "progress": progress,
                "unit": self._unit,
                "wall_seconds": round(time.perf_counter() - self._started, 2),
                **train_metrics,
                "evaluation": evaluation,
            }
        )
        logger.info(
            "%s %s %d: eval mean score %.2f, max %d, win rate %.2f, ends %s",
            self._method,
            self._unit,
            progress,
            summary.mean_score,
            evaluation["max_score"],
            summary.win_rate,
            evaluation["end_reasons"],
        )
        return summary
