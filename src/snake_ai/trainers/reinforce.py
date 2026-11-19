"""REINFORCE (Monte-Carlo policy gradient) in plain PyTorch.

Collects complete episodes with the current stochastic policy, computes the
discounted return from every step to the episode end and increases the log
probability of each action in proportion to that return. Returns are
standardized per batch, which acts as a constant baseline; there is no learned
value function (that is what distinguishes it from the actor-critic methods).
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch

from snake_ai.core import GameConfig
from snake_ai.envs import SnakeEnv
from snake_ai.runs import RunDirectory, SnapshotMeta, SnapshotRecorder
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml
from snake_ai.trainers.networks import (
    Architecture,
    PolicyNetwork,
    TorchPolicy,
    load_network,
    save_network,
)

METHOD = "reinforce"
UNIT = "env_steps"


@dataclass(frozen=True, slots=True)
class ReinforceSettings:
    total_timesteps: int
    episodes_per_update: int = 16
    seed: int = 0
    snapshot_every: int = 500_000
    torch_threads: int = 4
    learning_rate: float = 0.001
    gamma: float = 0.99
    entropy_coef: float = 0.01
    hidden_sizes: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        for name in ("total_timesteps", "episodes_per_update", "snapshot_every", "torch_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")

    @classmethod
    def load(cls, path: Path) -> tuple[Self, EnvSettings]:
        data = load_toml(path)
        return from_table(cls, data["train"]), from_table(EnvSettings, data.get("env", {}))


@dataclass(slots=True)
class _Episode:
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    rewards: list[float]


class ReinforceTrainer:
    def __init__(
        self,
        settings: ReinforceSettings,
        env_settings: EnvSettings,
        game_config: GameConfig,
        evaluation: EvaluationSettings,
        run: RunDirectory,
    ) -> None:
        self._settings = settings
        self._env_settings = env_settings
        self._game_config = game_config
        self._recorder = SnapshotRecorder(
            run,
            method=METHOD,
            unit=UNIT,
            seed=settings.seed,
            game_config=game_config,
            env_settings=env_settings,
            evaluation=evaluation,
        )

    def train(self) -> None:
        settings = self._settings
        torch.set_num_threads(settings.torch_threads)
        torch.manual_seed(settings.seed)
        self._recorder.write_run_files(asdict(settings))
        envs = [
            SnakeEnv(
                self._game_config,
                local_awareness=self._env_settings.local_awareness,
                max_episode_steps=self._env_settings.max_episode_steps,
            )
            for _ in range(settings.episodes_per_update)
        ]
        # Seed each env once; later resets continue its own random stream.
        for index, env in enumerate(envs):
            env.reset(seed=settings.seed * 10_000 + index)
        architecture = Architecture("policy", envs[0].encoder.size, settings.hidden_sizes)
        network = PolicyNetwork(architecture)
        optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)

        every = settings.snapshot_every
        steps, last_snapshot = 0, -1
        last_batch: dict[str, float | None] = {}
        while True:
            # One snapshot per crossed threshold, even if a batch skips several.
            if last_snapshot < 0 or steps // every > last_snapshot // every:
                self._snapshot(network, architecture, steps, last_batch)
                last_snapshot = steps
            if steps >= settings.total_timesteps:
                break
            episodes = self._collect(envs, network)
            last_batch = self._update(episodes, optimizer)
            steps += sum(len(episode.rewards) for episode in episodes)
        if steps != last_snapshot:
            self._snapshot(network, architecture, steps, last_batch)

    def _collect(self, envs: list[SnakeEnv], network: PolicyNetwork) -> list[_Episode]:
        """Play one episode per env in lockstep, batching the forward passes."""
        episodes = [_Episode([], [], []) for _ in envs]
        observations = [env.reset()[0] for env in envs]
        active = list(range(len(envs)))
        while active:
            batch = torch.as_tensor(np.stack([observations[index] for index in active]))
            distribution = torch.distributions.Categorical(logits=network(batch))
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            entropies = distribution.entropy()
            still_active = []
            for position, index in enumerate(active):
                observation, reward, terminated, truncated, _ = envs[index].step(
                    int(actions[position])
                )
                episode = episodes[index]
                episode.log_probs.append(log_probs[position])
                episode.entropies.append(entropies[position])
                episode.rewards.append(float(reward))
                observations[index] = observation
                if not (terminated or truncated):
                    still_active.append(index)
            active = still_active
        return episodes

    def _update(
        self, episodes: list[_Episode], optimizer: torch.optim.Optimizer
    ) -> dict[str, float | None]:
        returns = torch.cat([self._returns(episode.rewards) for episode in episodes])
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack([value for episode in episodes for value in episode.log_probs])
        entropy = torch.stack([value for episode in episodes for value in episode.entropies])
        loss = -(log_probs * advantages).mean() - self._settings.entropy_coef * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {
            "train_episode_reward": float(np.mean([sum(e.rewards) for e in episodes])),
            "train_episode_length": float(np.mean([len(e.rewards) for e in episodes])),
        }

    def _returns(self, rewards: list[float]) -> torch.Tensor:
        returns = np.zeros(len(rewards), dtype=np.float32)
        running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + self._settings.gamma * running
            returns[index] = running
        return torch.from_numpy(returns)

    def _snapshot(
        self,
        network: PolicyNetwork,
        architecture: Architecture,
        steps: int,
        train_metrics: dict[str, float | None],
    ) -> None:
        network.eval()
        self._recorder.record(
            steps,
            TorchPolicy(network),
            lambda directory: save_network(directory, network, architecture),
            **train_metrics,
        )
        network.train()


def load_policy(directory: Path, game_config: GameConfig) -> tuple[TorchPolicy, SnapshotMeta]:
    meta = SnapshotMeta.load(directory)
    meta.check_compatible(game_config)
    return TorchPolicy(load_network(directory)), meta
