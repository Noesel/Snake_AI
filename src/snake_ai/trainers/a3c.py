"""A3C: asynchronous advantage actor-critic with PyTorch multiprocessing.

One actor-critic network lives in shared memory. Each worker process owns an
environment and a local copy of the network: it syncs the local weights, plays
up to ``n_steps`` moves, computes n-step returns and advantages, backpropagates
locally and applies its gradients to the shared network with a shared Adam
optimizer. Workers never wait for each other; only the short optimizer step
is guarded by a lock so a snapshot never sees a half-applied update.
"""

import copy
import ctypes
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import torch
import torch.multiprocessing as mp

from snake_ai.core import GameConfig, ObservationEncoder
from snake_ai.envs import SnakeEnv
from snake_ai.runs import RunDirectory, SnapshotMeta, SnapshotRecorder
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml
from snake_ai.trainers.networks import (
    ActorCriticNetwork,
    Architecture,
    TorchPolicy,
    load_network,
    save_network,
)

METHOD = "a3c"
UNIT = "env_steps"
_POLL_SECONDS = 0.05
_JOIN_SECONDS = 30.0

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class A3cSettings:
    total_timesteps: int
    workers: int = 16
    seed: int = 0
    snapshot_every: int = 500_000
    learning_rate: float = 0.0005
    gamma: float = 0.99
    n_steps: int = 20
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        for name in ("total_timesteps", "workers", "snapshot_every", "n_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")

    @classmethod
    def load(cls, path: Path) -> tuple[Self, EnvSettings]:
        data = load_toml(path)
        return from_table(cls, data["train"]), from_table(EnvSettings, data.get("env", {}))


@dataclass(frozen=True)
class _Shared:
    """Objects shared between the coordinator and all workers."""

    network: ActorCriticNetwork
    optimizer_state: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    """Adam ``step``, ``exp_avg``, ``exp_avg_sq`` per parameter, in shared memory."""
    lock: Any
    # Raw shared values; every write happens while holding ``lock``.
    env_steps: ctypes.c_longlong
    updates: ctypes.c_longlong
    episodes: ctypes.c_longlong
    reward_sum: ctypes.c_double
    length_sum: ctypes.c_double


def _shared_adam(network: ActorCriticNetwork, shared: _Shared, lr: float) -> torch.optim.Adam:
    """Adam over this process's view of the shared parameters, with shared state.

    The optimizer is built per process so its parameter objects are the ones
    whose ``.grad`` the worker sets; only the state tensors are shared.
    """
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for parameter, (step, exp_avg, exp_avg_sq) in zip(
        network.parameters(), shared.optimizer_state, strict=True
    ):
        optimizer.state[parameter] = {"step": step, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    return optimizer


def _worker(
    rank: int,
    shared: _Shared,
    settings: A3cSettings,
    architecture: Architecture,
    game_config: GameConfig,
    env_settings: EnvSettings,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(settings.seed * 1_000 + rank)
    optimizer = _shared_adam(shared.network, shared, settings.learning_rate)
    local = ActorCriticNetwork(architecture)
    env = SnakeEnv(
        game_config,
        local_awareness=env_settings.local_awareness,
        max_episode_steps=env_settings.max_episode_steps,
    )
    observation, _ = env.reset(seed=settings.seed * 1_000 + rank)
    episode_reward, episode_length = 0.0, 0

    while shared.env_steps.value < settings.total_timesteps:
        local.load_state_dict(shared.network.state_dict())
        log_probs, values, entropies, rewards = [], [], [], []
        episode_over = terminated = False
        for _ in range(settings.n_steps):
            logits, value = local(torch.as_tensor(observation).unsqueeze(0))
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            observation, reward, terminated, truncated, _ = env.step(int(action))
            log_probs.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
            values.append(value)
            rewards.append(float(reward))
            episode_reward += float(reward)
            episode_length += 1
            if terminated or truncated:
                episode_over = True
                break

        # Bootstrap from the critic unless the game itself ended.
        with torch.no_grad():
            running = 0.0 if terminated else float(local(torch.as_tensor(observation)[None])[1])
        returns = []
        for reward in reversed(rewards):
            running = reward + settings.gamma * running
            returns.append(running)
        returns_tensor = torch.tensor(returns[::-1], dtype=torch.float32)
        values_tensor = torch.cat(values)
        advantages = returns_tensor - values_tensor
        loss = (
            -(torch.cat(log_probs) * advantages.detach()).mean()
            + settings.value_coef * advantages.pow(2).mean()
            - settings.entropy_coef * torch.cat(entropies).mean()
        )
        local.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local.parameters(), settings.max_grad_norm)

        with shared.lock:
            for shared_parameter, local_parameter in zip(
                shared.network.parameters(), local.parameters(), strict=True
            ):
                shared_parameter.grad = local_parameter.grad
            optimizer.step()
            shared.updates.value += 1
            shared.env_steps.value += len(rewards)
            if episode_over:
                shared.episodes.value += 1
                shared.reward_sum.value += episode_reward
                shared.length_sum.value += episode_length

        if episode_over:
            observation, _ = env.reset()
            episode_reward, episode_length = 0.0, 0


class A3cTrainer:
    def __init__(
        self,
        settings: A3cSettings,
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
        self.final_updates = 0
        self.final_optimizer_steps = 0

    def train(self) -> None:
        settings = self._settings
        self._recorder.write_run_files(asdict(settings))
        torch.manual_seed(settings.seed)
        encoder = ObservationEncoder(
            self._game_config, local_awareness=self._env_settings.local_awareness
        )
        architecture = Architecture("actor_critic", encoder.size, settings.hidden_sizes)
        network = ActorCriticNetwork(architecture)
        network.share_memory()
        context = mp.get_context("spawn")
        shared = _Shared(
            network=network,
            optimizer_state=[
                (
                    torch.zeros(()).share_memory_(),
                    torch.zeros_like(parameter).share_memory_(),
                    torch.zeros_like(parameter).share_memory_(),
                )
                for parameter in network.parameters()
            ],
            lock=context.Lock(),
            env_steps=context.Value("q", 0, lock=False),
            updates=context.Value("q", 0, lock=False),
            episodes=context.Value("q", 0, lock=False),
            reward_sum=context.Value("d", 0.0, lock=False),
            length_sum=context.Value("d", 0.0, lock=False),
        )
        self._snapshot(shared, architecture, progress=0)
        processes = [
            context.Process(
                target=_worker,
                args=(rank, shared, settings, architecture, self._game_config, self._env_settings),
                daemon=True,
            )
            for rank in range(settings.workers)
        ]
        for process in processes:
            process.start()
        try:
            self._coordinate(shared, architecture, processes)
        except BaseException:
            # Interrupted or failed: stop workers now instead of letting them
            # run to the step budget.
            for process in processes:
                process.terminate()
            raise
        for process in processes:
            process.join(timeout=_JOIN_SECONDS)
            if process.is_alive():
                process.terminate()
        failed = [process.exitcode for process in processes if process.exitcode != 0]
        if failed:
            raise RuntimeError(f"A3C workers failed with exit codes {failed}")
        self._snapshot(shared, architecture, progress=shared.env_steps.value)
        self.final_updates = shared.updates.value
        self.final_optimizer_steps = int(shared.optimizer_state[0][0])

    def _coordinate(
        self, shared: _Shared, architecture: Architecture, processes: list[mp.Process]
    ) -> None:
        every = self._settings.snapshot_every
        last_snapshot = 0
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                return
            steps = shared.env_steps.value
            if steps // every > last_snapshot // every and steps < self._settings.total_timesteps:
                self._snapshot(shared, architecture, steps)
                last_snapshot = steps
            time.sleep(_POLL_SECONDS)

    def _snapshot(self, shared: _Shared, architecture: Architecture, progress: int) -> None:
        with shared.lock:
            network = copy.deepcopy(shared.network)
            episodes = shared.episodes.value
            metrics: dict[str, float | None] = {
                "train_episode_reward": shared.reward_sum.value / episodes if episodes else None,
                "train_episode_length": shared.length_sum.value / episodes if episodes else None,
                "updates": float(shared.updates.value),
            }
            shared.episodes.value = 0
            shared.reward_sum.value = 0.0
            shared.length_sum.value = 0.0
        network.eval()
        self._recorder.record(
            progress,
            TorchPolicy(network),
            lambda directory: save_network(directory, network, architecture),
            **metrics,
        )


def load_policy(directory: Path, game_config: GameConfig) -> tuple[TorchPolicy, SnapshotMeta]:
    meta = SnapshotMeta.load(directory)
    meta.check_compatible(game_config)
    return TorchPolicy(load_network(directory)), meta
