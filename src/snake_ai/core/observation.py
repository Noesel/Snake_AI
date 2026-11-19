"""Shared observation encoding for every training method."""

import numpy as np
import numpy.typing as npt

from snake_ai.core.config import GameConfig
from snake_ai.core.types import Direction, GameState

type Observation = npt.NDArray[np.float32]


class ObservationEncoder:
    """Encode a state as a flat ``float32`` vector with values in ``[0, 1]``.

    Layout (version 2):

    1. Body-order grid, row-major: 0 free, tail 1 ... head L, divided by the
       number of cells.
    2. Food offset from the head in the snake's frame: cells ahead and cells
       to the right, each mapped from ``[-span, span]`` to ``[0, 1]`` with
       ``span = max(width, height)``; 0.5 means same row or column. Version 1
       used absolute food coordinates, which PPO failed to relate to the
       relative actions.
    3. Heading as one-hot ``UP, RIGHT, DOWN, LEFT``.
    4. Energy divided by the maximum energy.
    5. Optional local awareness: danger straight, left, right as 0 or 1.

    Snapshots must store ``VERSION``, the board size and ``local_awareness``;
    a model only fits observations produced with the same three values.
    """

    VERSION = 2

    def __init__(self, config: GameConfig, *, local_awareness: bool = False) -> None:
        self._config = config
        self._local_awareness = local_awareness
        self._cells = config.width * config.height
        self._size = self._cells + 2 + len(Direction) + 1 + (3 if local_awareness else 0)
        self._span = max(config.width, config.height)
        # Body values for every possible length, so encoding avoids a division.
        self._order = np.arange(self._cells, 0, -1, dtype=np.float32) / self._cells

    @property
    def size(self) -> int:
        return self._size

    @property
    def local_awareness(self) -> bool:
        return self._local_awareness

    def encode(self, state: GameState) -> Observation:
        width = self._config.width
        cells = self._cells
        observation = np.zeros(self._size, dtype=np.float32)

        length = len(state.body)
        indices = [row * width + column for column, row in state.body]
        # Head first: values L/cells down to 1/cells.
        observation[indices] = self._order[cells - length :]

        ahead, right = self._food_offset(state)
        observation[cells] = (ahead / self._span + 1) / 2
        observation[cells + 1] = (right / self._span + 1) / 2
        offset = cells + 2
        observation[offset + state.direction] = 1.0
        offset += len(Direction)
        observation[offset] = max(state.energy, 0) / self._config.max_energy
        if self._local_awareness:
            observation[offset + 1 : offset + 4] = self._dangers(state)
        return observation

    @staticmethod
    def _food_offset(state: GameState) -> tuple[int, int]:
        """Cells ahead and to the right of the head; ``(0, 0)`` without food."""
        if state.food is None:
            return 0, 0
        offset_column = state.food[0] - state.head[0]
        offset_row = state.food[1] - state.head[1]
        delta_column, delta_row = state.direction.delta
        ahead = offset_column * delta_column + offset_row * delta_row
        # Clockwise normal of the heading, i.e. the snake's right-hand side.
        right = offset_row * delta_column - offset_column * delta_row
        return ahead, right

    def _dangers(self, state: GameState) -> tuple[float, float, float]:
        # Same rule as SnakeGame.step: the tail cell is free on the next move.
        blocked = set(state.body[:-1])
        head_column, head_row = state.head
        dangers = []
        for turn in (0, -1, 1):
            delta_column, delta_row = Direction((state.direction + turn) % 4).delta
            column, row = head_column + delta_column, head_row + delta_row
            outside = not (0 <= column < state.width and 0 <= row < state.height)
            dangers.append(float(outside or (column, row) in blocked))
        return dangers[0], dangers[1], dangers[2]
