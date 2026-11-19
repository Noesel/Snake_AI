"""Train -> snapshot -> load -> evaluate for the plain PyTorch trainers."""

from pathlib import Path

from snake_ai.core import GameConfig
from snake_ai.evaluation import evaluate
from snake_ai.runs import RunDirectory
from snake_ai.settings import EnvSettings, EvaluationSettings
from snake_ai.trainers import reinforce


def test_reinforce_short_run_saves_loadable_snapshots(tmp_path: Path) -> None:
    settings = reinforce.ReinforceSettings(
        total_timesteps=300, episodes_per_update=4, snapshot_every=100, torch_threads=1
    )
    evaluation = EvaluationSettings(episodes=2, max_steps=50)
    run = RunDirectory.create(tmp_path, "reinforce")
    reinforce.ReinforceTrainer(settings, EnvSettings(), GameConfig(), evaluation, run).train()

    snapshots = sorted((run.path / "snapshots").iterdir())
    progresses = [int(path.name.rsplit("_", 1)[1]) for path in snapshots]
    assert progresses[0] == 0 and progresses[-1] >= 300
    assert len(progresses) == len(set(progresses))

    policy, meta = reinforce.load_policy(snapshots[-1], GameConfig())
    assert meta.method == "reinforce"
    first = evaluate(policy, GameConfig(), EnvSettings(), evaluation)
    assert first == evaluate(policy, GameConfig(), EnvSettings(), evaluation)
