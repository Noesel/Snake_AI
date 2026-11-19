"""Small PyTorch networks and their snapshot format for REINFORCE and A3C."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from snake_ai.core import Action, Observation

POLICY_FILE = "policy.pt"


@dataclass(frozen=True, slots=True)
class Architecture:
    """Enough to rebuild a network before loading its weights."""

    kind: str
    """``"policy"`` (actor only) or ``"actor_critic"`` (shared body, two heads)."""
    input_size: int
    hidden_sizes: tuple[int, ...]
    output_size: int = len(Action)


class PolicyNetwork(nn.Module):
    """MLP mapping observations to action logits."""

    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        self.body = _mlp_body(architecture.input_size, architecture.hidden_sizes)
        self.head = nn.Linear(architecture.hidden_sizes[-1], architecture.output_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(observations))

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self(observations)


class ActorCriticNetwork(nn.Module):
    """Shared MLP body with action-logit and state-value heads."""

    def __init__(self, architecture: Architecture) -> None:
        super().__init__()
        self.body = _mlp_body(architecture.input_size, architecture.hidden_sizes)
        self.actor = nn.Linear(architecture.hidden_sizes[-1], architecture.output_size)
        self.critic = nn.Linear(architecture.hidden_sizes[-1], 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observations)
        return self.actor(features), self.critic(features).squeeze(-1)

    def logits(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(self.body(observations))


type Network = PolicyNetwork | ActorCriticNetwork


def build(architecture: Architecture) -> Network:
    if architecture.kind == "policy":
        return PolicyNetwork(architecture)
    if architecture.kind == "actor_critic":
        return ActorCriticNetwork(architecture)
    raise ValueError(f"unknown network kind {architecture.kind!r}")


def save_network(directory: Path, network: Network, architecture: Architecture) -> None:
    torch.save(
        {"architecture": asdict(architecture), "state_dict": network.state_dict()},
        directory / POLICY_FILE,
    )


def load_network(directory: Path) -> Network:
    payload = torch.load(directory / POLICY_FILE, map_location="cpu", weights_only=True)
    spec = payload["architecture"]
    architecture = Architecture(**{**spec, "hidden_sizes": tuple(spec["hidden_sizes"])})
    network = build(architecture)
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network


class TorchPolicy:
    """Deterministic (argmax) playback of a PyTorch network."""

    def __init__(self, network: Network) -> None:
        self._network = network

    def act(self, observation: Observation) -> Action:
        with torch.no_grad():
            logits = self._network.logits(torch.as_tensor(observation).unsqueeze(0))
        return Action(int(logits.argmax(dim=-1).item()))


def _mlp_body(input_size: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    if not hidden_sizes:
        raise ValueError("at least one hidden layer is required")
    layers: list[nn.Module] = []
    previous = input_size
    for size in hidden_sizes:
        layers += [nn.Linear(previous, size), nn.Tanh()]
        previous = size
    return nn.Sequential(*layers)
