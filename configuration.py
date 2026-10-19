"""Serializable TensorNEAT settings, independent of JAX device initialization."""

from dataclasses import dataclass
import json
import math


TENSORNEAT_REVISION = "ee204900e8e55ffe0ed756651ea871c70230dd34"


@dataclass(frozen=True)
class TrainingConfig:
    population: int = 256
    episodes: int = 5
    max_steps: int = 10000
    seed: int = 42
    max_nodes: int = 16
    max_conns: int = 48
    species_size: int = 10
    max_stagnation: int = 1000
    fitness_target: float = 10000
    batch_size: int | None = None

    def validate(self):
        for name in ('population', 'episodes', 'max_steps', 'max_nodes', 'max_conns',
                     'species_size', 'max_stagnation'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.population < 2 or self.max_nodes < 6 or self.max_conns < 5:
            raise ValueError('Need population >= 2, max_nodes >= 6, max_conns >= 5')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if not isinstance(self.fitness_target, (int, float)) or not math.isfinite(self.fitness_target):
            raise ValueError('fitness_target must be finite')
        if self.batch_size is not None:
            if type(self.batch_size) is not int or self.batch_size < 1 or self.population % self.batch_size:
                raise ValueError('batch_size must be positive and divide population evenly')
        return self


def load_config(path):
    return TrainingConfig(**json.loads(path.read_text())).validate()
