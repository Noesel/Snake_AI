"""Train -> snapshot -> load -> evaluate for the PyTorch and NEAT trainers."""

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


def test_neat_short_run_saves_loadable_snapshots(tmp_path: Path) -> None:
    from snake_ai.trainers import neat_trainer

    settings = neat_trainer.NeatSettings(
        neat_config="configs/neat.ini", generations=3, episodes=2, snapshot_every=2, workers=2
    )
    evaluation = EvaluationSettings(episodes=2, max_steps=50)
    run = RunDirectory.create(tmp_path, "neat")
    env = EnvSettings(max_episode_steps=200)
    neat_trainer.NeatTrainer(settings, env, GameConfig(), evaluation, run).train()

    snapshots = sorted((run.path / "snapshots").iterdir())
    assert [path.name for path in snapshots] == [
        "generations_000000000000",
        "generations_000000000002",
    ]
    policy, meta = neat_trainer.load_policy(snapshots[-1], GameConfig())
    assert meta.unit == "generations"
    first = evaluate(policy, GameConfig(), env, evaluation)
    assert first == evaluate(policy, GameConfig(), env, evaluation)


def test_a3c_workers_update_shared_model(tmp_path: Path) -> None:
    """Gradients and optimizer state from every worker reach the shared model."""
    from snake_ai.trainers import a3c

    evaluation = EvaluationSettings(episodes=2, max_steps=50)
    for workers in (1, 3):
        settings = a3c.A3cSettings(
            total_timesteps=600, workers=workers, snapshot_every=10_000, n_steps=10
        )
        run = RunDirectory.create(tmp_path / str(workers), "a3c")
        trainer = a3c.A3cTrainer(settings, EnvSettings(), GameConfig(), evaluation, run)
        trainer.train()

        # One shared Adam step per worker update, from all processes.
        assert trainer.final_updates >= 600 // 10
        assert trainer.final_optimizer_steps == trainer.final_updates
        first, last = sorted((run.path / "snapshots").iterdir())
        initial, _ = a3c.load_policy(first, GameConfig())
        trained, meta = a3c.load_policy(last, GameConfig())
        assert meta.progress >= 600
        initial_weights = initial._network.state_dict()
        trained_weights = trained._network.state_dict()
        assert any(
            not initial_weights[name].equal(trained_weights[name]) for name in initial_weights
        )
