"""Contract between trained models and evaluation, demo and replay."""

from typing import Protocol

from snake_ai.core import Action, Observation


class Policy(Protocol):
    """Pick an action for an observation; deterministic for playback."""

    def act(self, observation: Observation) -> Action: ...
