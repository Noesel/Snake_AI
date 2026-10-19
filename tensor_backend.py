"""TensorNEAT evolution and compiled population/episode evaluation."""

import jax
import jax.numpy as jnp
from tensorneat.algorithm import NEAT
from tensorneat.common import ACT, AGG, State
from tensorneat.genome import BiasNode, DefaultConn, DefaultGenome
from tensorneat.genome.operations import DefaultMutation
from tensorneat.problem import BaseProblem

import jax_snake as snake


class SnakeProblem(BaseProblem):
    jitable = True

    def __init__(self, episodes, max_steps, seed):
        self.episodes, self.max_steps, self.seed = episodes, max_steps, seed

    @property
    def input_shape(self):
        return (5,)

    @property
    def output_shape(self):
        return (1,)

    def setup(self, state=State()):
        # A separate PRNG stream keeps environment draws out of evolution.
        key = jax.random.fold_in(jax.random.PRNGKey(self.seed), 0x534E414B)
        return state.register(snake_episode_keys=jax.random.split(key, self.episodes))

    def details(self, state, act_func, params):
        def episode(key):
            def continuing(game):
                return ~game.done & (game.steps < self.max_steps)

            def advance(game):
                output = act_func(state, params, snake.observe(game))[0]
                return snake.step(game, output)

            result = jax.lax.while_loop(continuing, advance, snake.reset(key))
            return result.score, result.steps

        scores, steps = jax.vmap(episode)(state.snake_episode_keys)
        return (jnp.mean(scores * 100.0 - 10.0), jnp.mean(scores.astype(jnp.float32)),
                jnp.max(scores), jnp.sum(steps))

    def evaluate(self, state, randkey, act_func, params):
        return self.details(state, act_func, params)[0]


class TensorBackend:
    def __init__(self, config):
        self.config = config.validate()
        genome = DefaultGenome(
            num_inputs=5, num_outputs=1, max_nodes=config.max_nodes, max_conns=config.max_conns,
            node_gene=BiasNode(activation_options=ACT.tanh, aggregation_options=AGG.sum,
                               bias_mutate_power=0.5, bias_mutate_rate=0.7, bias_replace_rate=0.1,
                               bias_lower_bound=-30, bias_upper_bound=30,
                               activation_replace_rate=0, aggregation_replace_rate=0),
            conn_gene=DefaultConn(weight_mutate_power=0.5, weight_mutate_rate=0.8,
                                  weight_replace_rate=0.1, weight_lower_bound=-30, weight_upper_bound=30),
            mutation=DefaultMutation(conn_add=0.5, conn_delete=0.5, node_add=0.2, node_delete=0.2),
            output_transform=ACT.tanh,
        )
        species = min(config.species_size, max(1, config.population // 2))
        self.algorithm = NEAT(genome=genome, pop_size=config.population, species_size=species,
                              max_stagnation=config.max_stagnation, species_elitism=min(2, species),
                              genome_elitism=2, survival_threshold=0.2, compatibility_threshold=3.0,
                              pop_batch_size=config.batch_size)
        self.problem = SnakeProblem(config.episodes, config.max_steps, config.seed)
        self.compiled_step = None

    def setup(self):
        state = self.algorithm.setup(State(randkey=jax.random.PRNGKey(self.config.seed)))
        return self.problem.setup(state)

    def evaluate(self, state):
        population = self.algorithm.ask(state)

        def batch(pop):
            transformed = jax.vmap(self.algorithm.transform, in_axes=(None, 0))(state, pop)
            return jax.vmap(self.problem.details, in_axes=(None, None, 0))(
                state, self.algorithm.forward, transformed)

        size = self.config.batch_size
        if size is None or size >= self.config.population:
            return batch(population)
        batches = jax.tree.map(lambda array: array.reshape(-1, size, *array.shape[1:]), population)
        _, results = jax.lax.scan(lambda carry, pop: (carry, batch(pop)), None, batches)
        return jax.tree.map(lambda array: array.reshape(-1), results)

    def generation(self, state):
        population = self.algorithm.ask(state)
        statistics = self.evaluate(state)
        next_state = self.algorithm.tell(state, statistics[0])
        return next_state, population, statistics

    def compile(self, state):
        # Explicit compilation keeps startup time out of measured generation time.
        self.compiled_step = jax.jit(self.generation).lower(state).compile()
        return self.compiled_step
