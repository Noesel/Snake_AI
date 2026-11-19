"""Headless Snake rules on a cell grid."""

import random
from collections import deque

from snake_ai.core.config import GameConfig
from snake_ai.core.types import Action, Cell, Direction, EndReason, GameState, StepResult

SIMULATION_VERSION = 1
"""Bump whenever a rule change can alter the outcome of a recorded replay."""

_FOOD_SAMPLING_ATTEMPTS = 64


class SnakeGame:
    """One Snake game: ``reset(seed)``, then ``step(action)`` until it is done.

    Food placement is the only random event and uses a generator owned by the
    game, so a seed and an action sequence reproduce an episode exactly.
    """

    def __init__(self, config: GameConfig) -> None:
        self._config = config
        self._rng = random.Random()
        self._body: deque[Cell] = deque()
        self._occupied: set[Cell] = set()
        self._food: Cell | None = None
        self._direction = Direction.UP
        self._energy = 0
        self._score = 0
        self._steps = 0
        self._end_reason: EndReason | None = None
        self._started = False
        self._state_cache: GameState | None = None

    @property
    def config(self) -> GameConfig:
        return self._config

    @property
    def state(self) -> GameState:
        """Immutable view of the current state, rebuilt at most once per move."""
        self._require_started()
        if self._state_cache is None:
            self._state_cache = GameState(
                width=self._config.width,
                height=self._config.height,
                body=tuple(self._body),
                food=self._food,
                direction=self._direction,
                energy=self._energy,
                score=self._score,
                steps=self._steps,
                end_reason=self._end_reason,
            )
        return self._state_cache

    def reset(self, seed: int) -> GameState:
        """Start the agreed initial position: centre, heading up, body below."""
        config = self._config
        column, head_row = config.width // 2, config.height // 2
        body = tuple((column, head_row + offset) for offset in range(config.start_length))
        return self._start(body, Direction.UP, config.max_energy, seed, food=None)

    def reset_to(self, state: GameState, seed: int) -> GameState:
        """Start from an explicit state, e.g. a test scenario.

        If ``state.food`` is ``None`` a new food cell is drawn from ``seed``.
        """
        if (state.width, state.height) != (self._config.width, self._config.height):
            raise ValueError("state board size differs from the game config")
        if state.done:
            raise ValueError("cannot start from a finished state")
        return self._start(state.body, state.direction, state.energy, seed, food=state.food)

    def step(self, action: Action) -> StepResult:
        """Apply one relative action and return its reward and end reason."""
        self._require_started()
        if self._end_reason is not None:
            raise RuntimeError("game is over; call reset() first")
        self._state_cache = None
        self._steps += 1
        rewards = self._config.rewards

        self._direction = self._direction.turned(action)
        delta_column, delta_row = self._direction.delta
        head_column, head_row = self._body[0]
        new_head = (head_column + delta_column, head_row + delta_row)

        if not self._inside(new_head):
            return self._end(EndReason.WALL, rewards.collision)
        tail = self._body[-1]
        # The tail moves away in the same step, so its cell is free. Food is
        # never on the body, so eating cannot keep the tail on the new head.
        if new_head in self._occupied and new_head != tail:
            return self._end(EndReason.SELF, rewards.collision)

        ate_food = new_head == self._food
        if not ate_food:
            self._occupied.discard(self._body.pop())
        self._body.appendleft(new_head)
        self._occupied.add(new_head)
        self._energy -= 1

        if ate_food:
            self._score += 1
            self._energy = self._config.max_energy
            self._food = self._place_food()
            if self._food is None:
                return self._end(EndReason.BOARD_FULL, rewards.food + rewards.board_full_bonus)
            return StepResult(rewards.food, ate_food=True, end_reason=None)
        if self._energy <= 0:
            return self._end(EndReason.ENERGY, rewards.energy_depleted)
        return StepResult(rewards.step, ate_food=False, end_reason=None)

    def _start(
        self,
        body: tuple[Cell, ...],
        direction: Direction,
        energy: int,
        seed: int,
        food: Cell | None,
    ) -> GameState:
        if not body or len(set(body)) != len(body) or not all(map(self._inside, body)):
            raise ValueError("body must be non-empty, unique and inside the board")
        if food is not None and (food in body or not self._inside(food)):
            raise ValueError("food must be a free cell inside the board")
        self._rng.seed(seed)
        self._body = deque(body)
        self._occupied = set(body)
        self._direction = direction
        self._energy = energy
        self._score = 0
        self._steps = 0
        self._end_reason = None
        self._started = True
        self._state_cache = None
        self._food = food if food is not None else self._place_food()
        if self._food is None:
            raise ValueError("no free cell for food")
        return self.state

    def _end(self, reason: EndReason, reward: float) -> StepResult:
        self._end_reason = reason
        return StepResult(reward, ate_food=reason is EndReason.BOARD_FULL, end_reason=reason)

    def _place_food(self) -> Cell | None:
        width, height = self._config.width, self._config.height
        if len(self._occupied) >= width * height:
            return None
        # Rejection sampling is fast on sparse boards; dense boards fall back
        # to an explicit list so placement always terminates.
        for _ in range(_FOOD_SAMPLING_ATTEMPTS):
            cell = (self._rng.randrange(width), self._rng.randrange(height))
            if cell not in self._occupied:
                return cell
        free = [
            (column, row)
            for row in range(height)
            for column in range(width)
            if (column, row) not in self._occupied
        ]
        return self._rng.choice(free)

    def _inside(self, cell: Cell) -> bool:
        column, row = cell
        return 0 <= column < self._config.width and 0 <= row < self._config.height

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("call reset() before using the game")
