"""A2C and PPO through Stable-Baselines3 on the Gymnasium adapter."""

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from snake_ai.core import Action, GameConfig, Observation
from snake_ai.envs import SnakeEnv
from snake_ai.evaluation import evaluate
from snake_ai.runs import RunDirectory, SnapshotMeta, environment_versions
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "a2c": A2C}
MODEL_FILE = "model.zip"
UNIT = "env_steps"

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Sb3Settings:
    algorithm: str
    total_timesteps: int
    n_envs: int = 8
    seed: int = 0
    snapshot_every: int = 50_000
    torch_threads: int = 4
    device: str = "cpu"
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    """Passed unchanged to the SB3 constructor; SB3 validates them."""

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}")
        for name in ("total_timesteps", "n_envs", "snapshot_every", "torch_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def load(cls, path: Path) -> tuple[Self, EnvSettings]:
        """Read ``[train]``, ``[algorithm]`` and ``[env]`` from a TOML file."""
        data = load_toml(path)
        settings = from_table(cls, {**data["train"], "hyperparameters": data.get("algorithm", {})})
        return settings, from_table(EnvSettings, data.get("env", {}))


class Sb3Trainer:
    """Train one SB3 model, saving an initial and periodic evaluated snapshots."""

    def __init__(
        self,
        settings: Sb3Settings,
        env_settings: EnvSettings,
        game_config: GameConfig,
        evaluation: EvaluationSettings,
        run: RunDirectory,
    ) -> None:
        self._settings = settings
        self._env_settings = env_settings
        self._game_config = game_config
        self._evaluation = evaluation
        self._run = run

    def train(self) -> None:
        settings = self._settings
        torch.set_num_threads(settings.torch_threads)
        self._run.write_json(
            "config.json",
            {
                "method": settings.algorithm,
                "game": asdict(self._game_config),
                "env": asdict(self._env_settings),
                "train": asdict(settings),
                "evaluation": asdict(self._evaluation),
            },
        )
        self._run.write_json("versions.json", environment_versions())

        env = make_vec_env(
            lambda: SnakeEnv(
                self._game_config,
                local_awareness=self._env_settings.local_awareness,
                max_episode_steps=self._env_settings.max_episode_steps,
            ),
            n_envs=settings.n_envs,
            seed=settings.seed,
        )
        algorithm = ALGORITHMS[settings.algorithm]
        model = algorithm(
            "MlpPolicy",
            env,
            seed=settings.seed,
            device=settings.device,
            verbose=0,
            **settings.hyperparameters,
        )
        started = time.perf_counter()
        callback = _SnapshotCallback(self.save_snapshot, settings.snapshot_every, started)
        self.save_snapshot(model, 0, started)
        model.learn(settings.total_timesteps, callback=callback)
        if model.num_timesteps != callback.last_snapshot:
            self.save_snapshot(model, model.num_timesteps, started)
        env.close()

    def save_snapshot(self, model: BaseAlgorithm, progress: int, started: float) -> None:
        directory = self._run.snapshot_dir(progress, UNIT)
        model.save(directory / MODEL_FILE)
        SnapshotMeta.describe(
            method=self._settings.algorithm,
            run=self._run.name,
            progress=progress,
            unit=UNIT,
            seed=self._settings.seed,
            game_config=self._game_config,
            local_awareness=self._env_settings.local_awareness,
        ).save(directory)

        summary = evaluate(
            Sb3Policy(model), self._game_config, self._env_settings, self._evaluation
        )
        elapsed = time.perf_counter() - started
        episodes = list(model.ep_info_buffer or [])
        record = {
            "progress": progress,
            "unit": UNIT,
            "wall_seconds": round(elapsed, 2),
            "train_episode_reward": _mean(info["r"] for info in episodes),
            "train_episode_length": _mean(info["l"] for info in episodes),
            "evaluation": summary.to_dict(),
        }
        self._run.append_metrics(record)
        logger.info(
            "%s %d: eval mean score %.2f, max %d, win rate %.2f, ends %s",
            UNIT,
            progress,
            summary.mean_score,
            record["evaluation"]["max_score"],
            summary.win_rate,
            record["evaluation"]["end_reasons"],
        )


class _SnapshotCallback(BaseCallback):
    def __init__(
        self, save: Callable[[BaseAlgorithm, int, float], None], every: int, started: float
    ) -> None:
        super().__init__()
        self._save = save
        self._every = every
        self._started = started
        self.last_snapshot = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_snapshot >= self._every:
            self.last_snapshot = self.num_timesteps
            self._save(self.model, self.num_timesteps, self._started)
        return True


class Sb3Policy:
    """Deterministic playback of an SB3 model."""

    def __init__(self, model: BaseAlgorithm) -> None:
        self._model = model

    def act(self, observation: Observation) -> Action:
        action, _ = self._model.predict(observation, deterministic=True)
        return Action(int(action))


def load_policy(directory: Path, game_config: GameConfig) -> tuple[Sb3Policy, SnapshotMeta]:
    meta = SnapshotMeta.load(directory)
    meta.check_compatible(game_config)
    model = ALGORITHMS[meta.method].load(directory / MODEL_FILE, device="cpu")
    return Sb3Policy(model), meta


def _mean(values: Any) -> float | None:
    items = list(values)
    return float(np.mean(items)) if items else None
