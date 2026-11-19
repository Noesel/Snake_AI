"""Draw game states with pygame. Import only when a window is wanted."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import pygame

from snake_ai.core import GameState

BACKGROUND = (40, 40, 40)
BOARD = (0, 0, 0)
GRID = (25, 25, 25)
HEAD = (220, 40, 40)
BODY_NEAR = (255, 255, 255)
BODY_FAR = (110, 110, 110)
FOOD = (40, 200, 70)
TEXT = (235, 235, 235)
MUTED = (160, 160, 160)
HIGHLIGHT = (240, 190, 40)

LINE_HEIGHT = 22
HEADER_HEIGHT = 3 * LINE_HEIGHT
FOOTER_HEIGHT = LINE_HEIGHT + 6
MARGIN = 12


@dataclass(frozen=True)
class Panel:
    """What one board shows: a state plus up to two text lines above it."""

    state: GameState
    title: str = ""
    subtitle: str = ""
    highlight: bool = False


class BoardRenderer:
    """Draw a single board with a three-line header at a given position."""

    def __init__(self, cell_size: int, font: pygame.font.Font) -> None:
        self._cell = cell_size
        self._font = font

    def size(self, state: GameState) -> tuple[int, int]:
        return state.width * self._cell, HEADER_HEIGHT + state.height * self._cell

    def draw(self, surface: pygame.Surface, panel: Panel, origin: tuple[int, int]) -> None:
        state, cell = panel.state, self._cell
        left, top = origin
        board_top = top + HEADER_HEIGHT
        self.text(surface, panel.title, (left, top))
        self.text(surface, panel.subtitle, (left, top + LINE_HEIGHT), MUTED)
        status = f"Score {state.score}  Energy {state.energy}  Steps {state.steps}"
        if state.end_reason is not None:
            status += f"  END {state.end_reason.value}"
        self.text(surface, status, (left, top + 2 * LINE_HEIGHT))

        board = pygame.Rect(left, board_top, state.width * cell, state.height * cell)
        pygame.draw.rect(surface, BOARD, board)
        if panel.highlight:
            pygame.draw.rect(surface, HIGHLIGHT, board.inflate(6, 6), width=3)
        for column in range(1, state.width):
            x = left + column * cell
            pygame.draw.line(surface, GRID, (x, board_top), (x, board.bottom - 1))
        for row in range(1, state.height):
            y = board_top + row * cell
            pygame.draw.line(surface, GRID, (left, y), (board.right - 1, y))

        if state.food is not None:
            center = self._center(state.food, left, board_top)
            pygame.draw.circle(surface, FOOD, center, cell * 0.4)
        length = len(state.body)
        # Fade towards the tail so the body order stays readable.
        for index, part in reversed(list(enumerate(state.body))):
            color = HEAD if index == 0 else _blend(BODY_NEAR, BODY_FAR, index / max(length - 1, 1))
            rect = pygame.Rect(0, 0, cell - 4, cell - 4)
            rect.center = self._center(part, left, board_top)
            pygame.draw.rect(surface, color, rect, border_radius=cell // 4)

    def _center(self, cell: tuple[int, int], left: int, top: int) -> tuple[int, int]:
        column, row = cell
        return (
            left + column * self._cell + self._cell // 2,
            top + row * self._cell + self._cell // 2,
        )

    def text(
        self,
        surface: pygame.Surface,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] = TEXT,
    ) -> None:
        if text:
            surface.blit(self._font.render(text, True, color), position)


class GameWindow:
    """A pygame window that shows several boards side by side.

    With ``footer=True`` a text line below the boards shows ``show(footer=...)``.
    """

    def __init__(
        self,
        caption: str,
        boards: int,
        board_size: tuple[int, int],
        cell_size: int,
        *,
        footer: bool = False,
    ):
        self._caption = caption
        self._boards = boards
        self._footer = footer
        self._cell = cell_size
        width, height = board_size
        self._board_pixels = width * cell_size, HEADER_HEIGHT + height * cell_size
        self._surface: pygame.Surface | None = None
        self._renderer: BoardRenderer | None = None
        self._clock: pygame.time.Clock | None = None

    def __enter__(self) -> Self:
        pygame.init()
        board_width, board_height = self._board_pixels
        size = (
            MARGIN + self._boards * (board_width + MARGIN),
            MARGIN * 2 + board_height + (FOOTER_HEIGHT if self._footer else 0),
        )
        self._surface = pygame.display.set_mode(size)
        pygame.display.set_caption(self._caption)
        self._renderer = BoardRenderer(self._cell, pygame.font.SysFont(None, 22))
        self._clock = pygame.time.Clock()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pygame.quit()

    def show(self, panels: Sequence[Panel], fps: int, footer: str = "") -> None:
        """Draw all panels and wait for the next frame; ``fps=0`` means no limit."""
        if self._surface is None or self._renderer is None or self._clock is None:
            raise RuntimeError("use GameWindow as a context manager")
        self._surface.fill(BACKGROUND)
        board_width = self._board_pixels[0]
        for index, panel in enumerate(panels):
            self._renderer.draw(
                self._surface, panel, (MARGIN + index * (board_width + MARGIN), MARGIN)
            )
        if self._footer:
            top = MARGIN + self._board_pixels[1] + FOOTER_HEIGHT - LINE_HEIGHT
            self._renderer.text(self._surface, footer, (MARGIN, top), MUTED)
        pygame.display.flip()
        if fps:
            self._clock.tick(fps)


def _blend(
    start: tuple[int, int, int], end: tuple[int, int, int], weight: float
) -> tuple[int, int, int]:
    return (
        round(start[0] + (end[0] - start[0]) * weight),
        round(start[1] + (end[1] - start[1]) * weight),
        round(start[2] + (end[2] - start[2]) * weight),
    )
