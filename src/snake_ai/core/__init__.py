"""Simulation core: rules, state and observation; no rendering or ML imports."""

from snake_ai.core.config import GameConfig, RewardConfig
from snake_ai.core.game import SIMULATION_VERSION, SnakeGame
from snake_ai.core.observation import Observation, ObservationEncoder
from snake_ai.core.types import Action, Cell, Direction, EndReason, GameState, StepResult

__all__ = [
    "SIMULATION_VERSION",
    "Action",
    "Cell",
    "Direction",
    "EndReason",
    "GameConfig",
    "GameState",
    "Observation",
    "ObservationEncoder",
    "RewardConfig",
    "SnakeGame",
    "StepResult",
]
