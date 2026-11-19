"""Value types shared by the simulation core, adapters and renderers."""

from dataclasses import dataclass
from enum import Enum, IntEnum

type Cell = tuple[int, int]
"""Board cell as ``(column, row)``; ``(0, 0)`` is the top-left cell."""


class Action(IntEnum):
    """Relative action; the integer value is the model output index."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Direction(IntEnum):
    """Absolute heading, ordered clockwise so turning is modular arithmetic."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Cell:
        """Column and row offset of one step in this direction."""
        return _DELTAS[self]

    def turned(self, action: Action) -> Direction:
        """Heading after applying a relative action."""
        return Direction((self + action - Action.STRAIGHT) % 4)


_DELTAS: dict[Direction, Cell] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}


class EndReason(Enum):
    """Why an episode terminated inside the game rules.

    An external step limit is not an end reason; adapters report it as
    truncation.
    """

    WALL = "wall"
    SELF = "self"
    ENERGY = "energy"
    BOARD_FULL = "board_full"


@dataclass(frozen=True, slots=True)
class GameState:
    """Immutable snapshot of a game, read by encoders and renderers."""

    width: int
    height: int
    body: tuple[Cell, ...]
    """Occupied cells from head to tail."""
    food: Cell | None
    """``None`` only after the board has been filled."""
    direction: Direction
    energy: int
    score: int
    steps: int
    end_reason: EndReason | None

    @property
    def head(self) -> Cell:
        return self.body[0]

    @property
    def done(self) -> bool:
        return self.end_reason is not None


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of one move."""

    reward: float
    ate_food: bool
    end_reason: EndReason | None

    @property
    def terminated(self) -> bool:
        return self.end_reason is not None
