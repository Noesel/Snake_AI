"""Command line entry point and composition root."""

import argparse
import importlib
import logging
from collections.abc import Sequence
from pathlib import Path

from snake_ai.core import Action, GameConfig, SnakeGame
from snake_ai.policies import Policy
from snake_ai.runs import RunDirectory, SnapshotMeta
from snake_ai.settings import EnvSettings, EvaluationSettings, from_table, load_toml

CONFIGS = Path("configs")
# Each module provides ``load_policy(directory, game_config)``.
TRAINER_MODULES = {
    "ppo": "snake_ai.trainers.sb3",
    "a2c": "snake_ai.trainers.sb3",
    "reinforce": "snake_ai.trainers.reinforce",
    "neat": "snake_ai.trainers.neat_trainer",
    "a3c": "snake_ai.trainers.a3c",
}
# Order of the boards in the demo, following the talk.
DEMO_METHODS = ("neat", "reinforce", "a2c", "a3c", "ppo")

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
    train.add_argument("method", choices=sorted(TRAINER_MODULES))
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

    demo = commands.add_parser("demo", help="show all methods side by side on the same games")
    demo.add_argument("--runs", type=Path, default=Path("runs"))
    demo.add_argument(
        "--methods",
        default=",".join(DEMO_METHODS),
        help="comma-separated boards, left to right (default: %(default)s)",
    )
    demo.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="METHOD=DIR",
        help="use this run instead of the newest compatible one; repeatable",
    )
    demo.add_argument(
        "--start",
        choices=("first", "last", "best"),
        default="last",
        help="initial snapshot per board; best = highest evaluation mean score",
    )
    demo.add_argument("--game", type=int, default=1, help="evaluation game to start with (1-based)")
    demo.add_argument("--fps", type=int, default=10)
    demo.add_argument("--cell-size", type=int, default=32)
    demo.set_defaults(handler=_demo)

    play = commands.add_parser("play", help="play manually to check rules and rendering")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--fps", type=int, default=6)
    play.add_argument("--cell-size", type=int, default=40)
    play.set_defaults(handler=_play)
    return parser


def _train(args: argparse.Namespace) -> None:
    config_path = args.config or CONFIGS / f"{args.method}.toml"
    game_config = GameConfig.load(args.game_config)
    evaluation = _evaluation_settings(args)
    if args.method == "a3c":
        from snake_ai.trainers.a3c import A3cSettings, A3cTrainer

        a3c_settings, env_settings = A3cSettings.load(config_path)
        A3cTrainer(a3c_settings, env_settings, game_config, evaluation, _create_run(args)).train()
        return
    if args.method == "neat":
        from snake_ai.trainers.neat_trainer import NeatSettings, NeatTrainer

        neat_settings, env_settings = NeatSettings.load(config_path)
        trainer = NeatTrainer(
            neat_settings, env_settings, game_config, evaluation, _create_run(args)
        )
        trainer.train()
        return
    if args.method == "reinforce":
        from snake_ai.trainers.reinforce import ReinforceSettings, ReinforceTrainer

        settings, env_settings = ReinforceSettings.load(config_path)
        run = _create_run(args)
        ReinforceTrainer(settings, env_settings, game_config, evaluation, run).train()
        return

    from snake_ai.trainers.sb3 import Sb3Settings, Sb3Trainer

    sb3_settings, env_settings = Sb3Settings.load(config_path)
    if sb3_settings.algorithm != args.method:
        raise SystemExit(f"config is for {sb3_settings.algorithm}, not {args.method}")
    run = _create_run(args)
    Sb3Trainer(sb3_settings, env_settings, game_config, evaluation, run).train()


def _create_run(args: argparse.Namespace) -> RunDirectory:
    run = RunDirectory.create(args.runs, args.method)
    logger.info("run directory %s", run.path)
    return run


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


DEMO_HELP = (
    "Left/Right snapshot  1-5/Tab board  N/P game  R restart  Space pause  +/- speed  Esc quit"
)
_AUTO_ADVANCE_SECONDS = 2.0


def _demo(args: argparse.Namespace) -> None:
    """Interactive side-by-side demo; see ``DEMO_HELP`` for the keys."""
    import pygame

    from snake_ai.demo import Demo, DemoSlot, latest_run, load_run, start_index
    from snake_ai.render.pygame_view import GameWindow, Panel

    game_config = GameConfig.load(args.game_config)
    evaluation = _evaluation_settings(args)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = set(methods) - set(TRAINER_MODULES)
    if unknown or not methods:
        raise SystemExit(f"unknown methods: {', '.join(sorted(unknown)) or '(none)'}")
    overrides = _run_overrides(args.run)

    def load(path: Path) -> tuple[Policy, SnapshotMeta]:
        return _load_policy(path, game_config)

    slots = []
    for method in methods:
        if method in overrides:
            entries = load_run(overrides[method], game_config)
        else:
            entries = latest_run(args.runs, method, game_config)
        if not entries:
            logger.warning("%s: no compatible snapshot found", method)
        else:
            logger.info("%s: %s (%d snapshots)", method, entries[0].meta.run, len(entries))
        index = start_index(entries, args.start)
        slots.append(DemoSlot(method, entries, load, game_config, evaluation.max_steps, index))
    demo = Demo(slots, evaluation.game_seeds(), game=args.game - 1)

    fps, paused, finished_frames = max(args.fps, 1), False, 0
    keys = {
        pygame.K_LEFT: lambda: demo.change_snapshot(-1),
        pygame.K_RIGHT: lambda: demo.change_snapshot(1),
        pygame.K_TAB: lambda: demo.cycle_slot(1),
        pygame.K_n: lambda: demo.change_game(1),
        pygame.K_p: lambda: demo.change_game(-1),
        pygame.K_r: demo.restart,
    }
    keys.update({pygame.K_1 + i: lambda i=i: demo.select_slot(i) for i in range(len(slots))})
    board = (game_config.width, game_config.height)
    caption = "Snake AI demo"
    with GameWindow(caption, len(slots), board, args.cell_size, footer=True) as window:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key == pygame.K_ESCAPE:
                    return
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    fps = min(fps * 2, 240)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(fps // 2, 1)
                elif event.key in keys:
                    keys[event.key]()
                    finished_frames = 0
            if not paused:
                demo.step()
                # Hold the finished boards briefly, then move on together.
                finished_frames = finished_frames + 1 if demo.all_done else 0
                if finished_frames >= _AUTO_ADVANCE_SECONDS * fps:
                    demo.change_game(1)
                    finished_frames = 0
            panels = [
                Panel(slot.state, slot.title(), slot.subtitle(), index == demo.selected)
                for index, slot in enumerate(demo.slots)
            ]
            status = f"{demo.game_label()}  {fps} fps{'  PAUSED' if paused else ''}"
            window.show(panels, fps, f"{status}   |   {DEMO_HELP}")


def _run_overrides(values: Sequence[str]) -> dict[str, Path]:
    overrides = {}
    for value in values:
        method, separator, directory = value.partition("=")
        if not separator or method not in TRAINER_MODULES:
            raise SystemExit(f"--run expects METHOD=DIR with a known method, got {value!r}")
        overrides[method] = Path(directory)
    return overrides


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
    if method not in TRAINER_MODULES:
        raise SystemExit(f"no loader for method {method!r}")
    module = importlib.import_module(TRAINER_MODULES[method])
    try:
        return module.load_policy(snapshot, game_config)
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
