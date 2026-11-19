"""Command line entry point and composition root."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from snake_ai.core import Action, GameConfig, SnakeGame
from snake_ai.policies import Policy
from snake_ai.runs import SnapshotMeta
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml

CONFIGS = Path("configs")
SB3_METHODS = ("ppo", "a2c")

logger = logging.getLogger("snake_ai")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snake-ai", description="Snake AI experiments")
    parser.add_argument("--game-config", type=Path, default=CONFIGS / "game.toml")
    parser.add_argument("--eval-config", type=Path, default=CONFIGS / "evaluation.toml")
    commands = parser.add_subparsers(required=True)

    train = commands.add_parser("train", help="train one method headless")
    train.add_argument("method", choices=SB3_METHODS)
    train.add_argument("--config", type=Path, help="method config (default: configs/<method>.toml)")
    train.add_argument("--runs", type=Path, default=Path("runs"))
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="evaluate a snapshot on the fixed seeds")
    evaluate.add_argument("snapshot", type=Path)
    evaluate.set_defaults(handler=_evaluate)

    watch = commands.add_parser("watch", help="show a snapshot playing new games")
    watch.add_argument("snapshot", type=Path)
    watch.add_argument("--seed", type=int, default=0)
    watch.add_argument("--fps", type=int, default=10)
    watch.add_argument("--cell-size", type=int, default=40)
    watch.set_defaults(handler=_watch)

    play = commands.add_parser("play", help="play manually to check rules and rendering")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--fps", type=int, default=6)
    play.add_argument("--cell-size", type=int, default=40)
    play.set_defaults(handler=_play)
    return parser


def _train(args: argparse.Namespace) -> None:
    from snake_ai.runs import RunDirectory
    from snake_ai.trainers.sb3 import Sb3Settings, Sb3Trainer

    settings, env_settings = Sb3Settings.load(args.config or CONFIGS / f"{args.method}.toml")
    if settings.algorithm != args.method:
        raise SystemExit(f"config is for {settings.algorithm}, not {args.method}")
    run = RunDirectory.create(args.runs, args.method)
    logger.info("run directory %s", run.path)
    game_config = GameConfig.load(args.game_config)
    Sb3Trainer(settings, env_settings, game_config, _evaluation_settings(args), run).train()


def _evaluate(args: argparse.Namespace) -> None:
    from snake_ai.evaluation import evaluate

    game_config = GameConfig.load(args.game_config)
    policy, meta = _load_policy(args.snapshot, game_config)
    env_settings = EnvSettings(local_awareness=meta.local_awareness)
    summary = evaluate(policy, game_config, env_settings, _evaluation_settings(args))
    logger.info("%s %s %d: %s", meta.method, meta.unit, meta.progress, summary.to_dict())


def _watch(args: argparse.Namespace) -> None:
    from snake_ai.core import ObservationEncoder
    from snake_ai.render.pygame_view import GameWindow, Panel

    game_config = GameConfig.load(args.game_config)
    policy, meta = _load_policy(args.snapshot, game_config)
    encoder = ObservationEncoder(game_config, local_awareness=meta.local_awareness)
    game = SnakeGame(game_config)
    seed = args.seed
    game.reset(seed)
    title = f"{meta.method.upper()}  |  {meta.progress:,} {meta.unit}  |  seed {seed}"
    board = (game_config.width, game_config.height)
    with GameWindow("Snake AI - close window to quit", 1, board, args.cell_size) as window:
        pause = 0
        while not _quit_requested():
            state = game.state
            if state.done and pause >= args.fps:
                seed, pause = seed + 1, 0
                state = game.reset(seed)
                title = f"{meta.method.upper()}  |  {meta.progress:,} {meta.unit}  |  seed {seed}"
            elif state.done:
                pause += 1
            else:
                game.step(policy.act(encoder.encode(state)))
            window.show([Panel(game.state, title)], args.fps)


def _play(args: argparse.Namespace) -> None:
    """Arrow keys or A/D turn relative to the heading; R restarts, Esc quits."""
    import pygame

    from snake_ai.render.pygame_view import GameWindow, Panel

    config = GameConfig.load(args.game_config)
    game = SnakeGame(config)
    seed = args.seed
    game.reset(seed)
    turns = {
        pygame.K_LEFT: Action.LEFT,
        pygame.K_a: Action.LEFT,
        pygame.K_RIGHT: Action.RIGHT,
        pygame.K_d: Action.RIGHT,
    }
    caption = "Snake AI - arrows/A-D turn, R restart, Esc quit"
    board = (config.width, config.height)
    with GameWindow(caption, 1, board, args.cell_size) as window:
        while True:
            action = Action.STRAIGHT
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_r:
                    seed += 1
                    game.reset(seed)
                elif event.key in turns and action is Action.STRAIGHT:
                    action = turns[event.key]
            if not game.state.done:
                game.step(action)
            window.show([Panel(game.state, f"Manual  |  seed {seed}")], args.fps)


def _load_policy(snapshot: Path, game_config: GameConfig) -> tuple[Policy, SnapshotMeta]:
    method = SnapshotMeta.load(snapshot).method
    if method not in SB3_METHODS:
        raise SystemExit(f"no loader for method {method!r}")
    from snake_ai.trainers.sb3 import load_policy

    try:
        return load_policy(snapshot, game_config)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def _evaluation_settings(args: argparse.Namespace) -> EvaluationSettings:
    return from_table(EvaluationSettings, load_toml(args.eval_config).get("evaluation", {}))


def _quit_requested() -> bool:
    import pygame

    return any(
        event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
        for event in pygame.event.get()
    )
