"""Train -> snapshot -> load -> evaluate; the chain every trainer must keep."""

from pathlib import Path

import pytest

from snake_ai.core import GameConfig
from snake_ai.evaluation import evaluate
from snake_ai.runs import RunDirectory
from snake_ai.settings import EnvSettings, EvaluationSettings
from snake_ai.trainers.sb3 import Sb3Settings, Sb3Trainer, load_policy


@pytest.mark.parametrize("algorithm", ["ppo", "a2c"])
def test_short_run_saves_loadable_snapshots(tmp_path: Path, algorithm: str) -> None:
    hyperparameters = {"n_steps": 64} | ({"batch_size": 64} if algorithm == "ppo" else {})
    settings = Sb3Settings(
        algorithm=algorithm,
        total_timesteps=256,
        n_envs=2,
        snapshot_every=128,
        torch_threads=1,
        hyperparameters=hyperparameters,
    )
    evaluation = EvaluationSettings(episodes=2, max_steps=50)
    run = RunDirectory.create(tmp_path, algorithm)
    Sb3Trainer(settings, EnvSettings(), GameConfig(), evaluation, run).train()

    snapshots = sorted((run.path / "snapshots").iterdir())
    assert [path.name for path in snapshots] == [
        "env_steps_000000000000",
        "env_steps_000000000128",
        "env_steps_000000000256",
    ]
    assert len((run.path / "metrics.jsonl").read_text().splitlines()) == 3

    policy, meta = load_policy(snapshots[-1], GameConfig())
    assert meta.progress == 256
    first = evaluate(policy, GameConfig(), EnvSettings(), evaluation)
    second = evaluate(policy, GameConfig(), EnvSettings(), evaluation)
    assert first == second


def test_snapshot_from_other_board_is_rejected(tmp_path: Path) -> None:
    settings = Sb3Settings(
        algorithm="ppo",
        total_timesteps=64,
        n_envs=1,
        snapshot_every=64,
        torch_threads=1,
        hyperparameters={"n_steps": 64, "batch_size": 64},
    )
    run = RunDirectory.create(tmp_path, "ppo")
    Sb3Trainer(settings, EnvSettings(), GameConfig(), EvaluationSettings(episodes=1), run).train()
    snapshot = next((run.path / "snapshots").iterdir())
    with pytest.raises(ValueError, match="board"):
        load_policy(snapshot, GameConfig(width=12, height=12))
