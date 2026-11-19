"""Command line entry point and composition root."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from snake_ai.core import Action, GameConfig, SnakeGame

DEFAULT_GAME_CONFIG = Path("configs/game.toml")


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.handler(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="snake-ai", description=__doc__)
    commands = parser.add_subparsers(required=True)

    play = commands.add_parser("play", help="play manually to check rules and rendering")
    play.add_argument("--config", type=Path, default=DEFAULT_GAME_CONFIG)
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--fps", type=int, default=6)
    play.add_argument("--cell-size", type=int, default=40)
    play.set_defaults(handler=_play)
    return parser


def _play(args: argparse.Namespace) -> None:
    """Arrow keys or A/D turn relative to the heading; R restarts, Esc quits."""
    import pygame

    from snake_ai.render.pygame_view import GameWindow, Panel

    config = GameConfig.load(args.config)
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
