"""A2C and PPO through Stable-Baselines3 on the Gymnasium adapter."""

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
from snake_ai.runs import RunDirectory, SnapshotMeta, SnapshotRecorder
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {"ppo": PPO, "a2c": A2C}
MODEL_FILE = "model.zip"
UNIT = "env_steps"


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
        recorder = SnapshotRecorder(
            self._run,
            method=settings.algorithm,
            unit=UNIT,
            seed=settings.seed,
            game_config=self._game_config,
            env_settings=self._env_settings,
            evaluation=self._evaluation,
        )
        recorder.write_run_files(asdict(settings))
        env = make_vec_env(
            lambda: SnakeEnv(
                self._game_config,
                local_awareness=self._env_settings.local_awareness,
                max_episode_steps=self._env_settings.max_episode_steps,
            ),
            n_envs=settings.n_envs,
            seed=settings.seed,
        )
        model = ALGORITHMS[settings.algorithm](
            "MlpPolicy",
            env,
            seed=settings.seed,
            device=settings.device,
            verbose=0,
            **settings.hyperparameters,
        )

        def snapshot(progress: int) -> None:
            episodes = list(model.ep_info_buffer or [])
            recorder.record(
                progress,
                Sb3Policy(model),
                lambda directory: model.save(directory / MODEL_FILE),
                train_episode_reward=_mean(info["r"] for info in episodes),
                train_episode_length=_mean(info["l"] for info in episodes),
            )

        callback = _SnapshotCallback(snapshot, settings.snapshot_every)
        snapshot(0)
        model.learn(settings.total_timesteps, callback=callback)
        if model.num_timesteps != callback.last_snapshot:
            snapshot(model.num_timesteps)
        env.close()


class _SnapshotCallback(BaseCallback):
    def __init__(self, snapshot: Callable[[int], None], every: int) -> None:
        super().__init__()
        self._snapshot = snapshot
        self._every = every
        self.last_snapshot = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_snapshot >= self._every:
            self.last_snapshot = self.num_timesteps
            self._snapshot(self.num_timesteps)
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
