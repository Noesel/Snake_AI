from dataclasses import replace
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from configuration import TrainingConfig, load_config


AVAILABLE = bool(importlib.util.find_spec('jax') and importlib.util.find_spec('tensorneat'))
if AVAILABLE:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jax_snake as snake
    from tensor_backend import TensorBackend
    from training import restore_checkpoint, save_checkpoint
    from evaluation import step_game
    from tests.support import make_game


class ConfigurationTests(unittest.TestCase):
    def test_checked_in_configuration_is_valid(self):
        config = load_config(Path(__file__).resolve().parents[1] / 'config-tensorneat.json')
        self.assertEqual(config.fitness_target, 10000)

    def test_invalid_capacities_and_batches_are_rejected(self):
        for values in ({'population': 1}, {'max_nodes': 5}, {'max_conns': 4}, {'episodes': 0},
                       {'seed': -1}, {'seed': 2**32}, {'fitness_target': float('inf')},
                       {'batch_size': 3}, {'batch_size': True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(TrainingConfig(), **values).validate()


@unittest.skipUnless(AVAILABLE, 'requires TensorNEAT and JAX')
class JaxSnakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.advance = staticmethod(jax.jit(snake.step))
        cls.observation = staticmethod(jax.jit(snake.observe))

    def from_reference(self, game):
        def position(point):
            return ((point[0] - game.start_point_x) // 10 - 1,
                    (point[1] - game.start_point_y - game.offset) // 10 - 1)
        positions = [position(point) for point in game.snake.body_parts][::-1]
        cells = np.array([y * snake.WIDTH + x for x, y in positions], dtype=np.int32)
        state = snake.reset(jax.random.PRNGKey(7))
        head = jnp.array(position((game.snake.x_head_pos, game.snake.y_head_pos)), dtype=jnp.int32)
        food = position((game.bait.x, game.bait.y))
        return state._replace(head=head, direction=jnp.int32(game.snake.direction // 90),
                              body=state.body.at[:len(cells)].set(cells),
                              length=jnp.int32(len(cells)),
                              occupied=jnp.zeros(snake.CAPACITY, dtype=bool).at[cells].set(True),
                              food=jnp.int32(food[1] * snake.WIDTH + food[0]),
                              energy=jnp.int32(game.snake.energy))

    def test_observations_match_reference_for_headings_edges_and_body(self):
        for origin in ((0, 20), (460, 580)):
            for heading in (0, 90, 180, 270):
                for head in ((10, 30), (210, 230), (100, 120)):
                    with self.subTest(origin=origin, heading=heading, head=head):
                        game = make_game(*origin)
                        game.snake.x_head_pos = origin[0] + head[0]
                        game.snake.y_head_pos = origin[1] + head[1]
                        game.snake.body_parts = [(game.snake.x_head_pos, game.snake.y_head_pos)]
                        game.snake.direction = heading
                        state = self.from_reference(game)
                        np.testing.assert_array_equal(self.observation(state), game.get_observation())
        for heading in (0, 90, 180, 270):
            game = make_game()
            game.snake.direction = heading
            game.snake.body_parts = [(100, 140), (100, 150), (110, 150), (110, 140), (120, 140)]
            np.testing.assert_array_equal(self.observation(self.from_reference(game)), game.get_observation())

    def test_tail_cell_is_safe_and_moves_consistently(self):
        game = make_game()
        game.snake.direction = 90
        game.snake.body_parts = [(100, 140), (100, 150), (110, 150), (110, 140)]
        state = self.from_reference(game)
        self.assertEqual(float(self.observation(state)[0]), 0)
        result = self.advance(state, 0.0)
        self.assertFalse(bool(result.collided))
        self.assertEqual(int(jnp.sum(result.occupied)), 4)

    def test_trajectories_match_reference_with_synchronized_food(self):
        for seed in (0, 1, 7):
            game = make_game()
            state = snake.reset(jax.random.PRNGKey(seed))
            # Match the starting food; subsequent food is synchronized after eating.
            for _ in range(120):
                food = int(state.food)
                game.bait.x = game.start_point_x + 10 + (food % snake.WIDTH) * 10
                game.bait.y = game.start_point_y + game.offset + 10 + (food // snake.WIDTH) * 10
                np.testing.assert_array_equal(self.observation(state), game.get_observation())
                front, left, right, forward_food, right_food = game.get_observation()
                if right_food > 0 and not right:
                    output = 1.0
                elif right_food < 0 and not left:
                    output = -1.0
                elif not front:
                    output = 0.0
                else:
                    output = -1.0 if not left else 1.0
                step_game(game, SimpleNamespace(activate=lambda observation: [output]))
                state = self.advance(state, output)
                self.assertEqual(int(state.energy), game.snake.energy)
                self.assertEqual(int(state.score), game.score)
                self.assertEqual(int(state.length), len(game.snake.body_parts))
                self.assertEqual(bool(state.collided), game.snake.collided)
                self.assertEqual(bool(state.done), game.done)
                expected_head = ((game.snake.x_head_pos - game.start_point_x) // 10 - 1,
                                 (game.snake.y_head_pos - game.start_point_y - game.offset) // 10 - 1)
                np.testing.assert_array_equal(state.head, expected_head)
                if game.done:
                    break

    def test_full_and_nearly_full_board_food_sampling(self):
        occupied = jnp.ones(snake.CAPACITY, dtype=bool)
        key = jax.random.PRNGKey(1)
        self.assertEqual(int(snake.sample_food(key, occupied)[1]), -1)
        self.assertEqual(int(snake.sample_food(key, occupied.at[123].set(False))[1]), 123)

    def test_growth_cannot_place_food_inside_the_body(self):
        state = snake.reset(jax.random.PRNGKey(3))
        state = state._replace(food=jnp.int32(10 * snake.WIDTH + 9))
        result = self.advance(state, 0.0)
        self.assertEqual(int(result.length), 4)
        self.assertEqual(int(result.energy), 39)
        self.assertEqual(int(result.score), 1)
        self.assertFalse(bool(result.occupied[result.food]))

    def test_finished_lanes_stay_frozen_in_a_batch(self):
        state = snake.reset(jax.random.PRNGKey(3))
        stopped = state._replace(done=jnp.array(True), energy=jnp.int32(0))
        batch = jax.tree.map(lambda live, dead: jnp.stack((live, dead)), state, stopped)
        result = jax.jit(jax.vmap(snake.step))(batch, jnp.array((0.0, 1.0)))
        self.assertEqual(int(result.steps[0]), 1)
        for actual, expected in zip(jax.tree.leaves(result), jax.tree.leaves(stopped)):
            np.testing.assert_array_equal(actual[1], expected)

    def test_eating_the_last_free_cell_completes_the_board(self):
        path = np.array([y * snake.WIDTH + x for y in range(snake.WIDTH)
                         for x in (range(snake.WIDTH) if y % 2 == 0 else range(snake.WIDTH - 1, -1, -1))])
        state = snake.reset(jax.random.PRNGKey(4))
        state = state._replace(head=jnp.array((19, 20)), direction=jnp.int32(1),
                              length=jnp.int32(snake.CAPACITY - 1),
                              body=state.body.at[:-1].set(path[:-1]),
                              occupied=jnp.ones(snake.CAPACITY, dtype=bool).at[path[-1]].set(False),
                              food=jnp.int32(path[-1]))
        result = self.advance(state, 0.0)
        self.assertTrue(bool(result.done))
        self.assertFalse(bool(result.collided))
        self.assertEqual(int(result.length), snake.CAPACITY)
        self.assertEqual(int(result.food), -1)


@unittest.skipUnless(AVAILABLE, 'requires TensorNEAT and JAX')
class TensorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = TrainingConfig(population=4, episodes=2, max_steps=30, max_nodes=8, max_conns=8)
        cls.backend = TensorBackend(cls.config)
        cls.initial = cls.backend.setup()
        cls.compiled = cls.backend.compile(cls.initial)

    def assert_states_equal(self, actual, expected):
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_allclose(left, right, equal_nan=True, rtol=0, atol=0)

    def test_batched_population_matches_individual_evaluation(self):
        backend, state = self.backend, self.initial
        expected = jax.jit(backend.evaluate)(state)
        population = backend.algorithm.ask(state)
        evaluate_one = jax.jit(lambda nodes, conns: backend.problem.details(
            state, backend.algorithm.forward, backend.algorithm.transform(state, (nodes, conns))))
        individual = [evaluate_one(population[0][i], population[1][i]) for i in range(4)]
        actual = jax.tree.map(lambda *values: jnp.stack(values), *individual)
        self.assert_states_equal(actual, expected)

    def test_memory_limited_batches_match_full_population(self):
        chunked = TensorBackend(replace(self.config, batch_size=2))
        actual = jax.jit(chunked.evaluate)(self.initial)
        expected = jax.jit(self.backend.evaluate)(self.initial)
        self.assert_states_equal(actual, expected)

    def test_checkpoint_resume_matches_uninterrupted_evolution(self):
        first, _, _ = self.compiled(self.initial)
        expected, _, _ = self.compiled(first)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pkl.gz'
            save_checkpoint(checkpoint, self.config, first, None)
            config, restored, best = restore_checkpoint(checkpoint)
            self.assertEqual(config, self.config)
            self.assertIsNone(best)
            actual, _, _ = self.compiled(restored)
        self.assert_states_equal(actual, expected)

    def test_legacy_checkpoint_format_is_rejected(self):
        from training import atomic_pickle
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'old.pkl.gz'
            atomic_pickle(checkpoint, {'version': 1})
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                restore_checkpoint(checkpoint)

    def test_cli_runs_full_budget_and_resumes_with_saved_best(self):
        from training import main
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            configuration = root / 'config.json'
            from dataclasses import asdict
            configuration.write_text(json.dumps(asdict(replace(self.config, fitness_target=-10))))
            output = main(['--headless', '--device', 'cpu', '--config', str(configuration),
                           '--generations', '2', '--output', str(root / 'first')])
            rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['generation'] for row in rows], [0, 1])
            checkpoint = output / 'checkpoint.pkl.gz'
            config, state, best = restore_checkpoint(checkpoint)
            best['fitness'] = 999999  # No new result can replace this fixture.
            save_checkpoint(checkpoint, config, state, best)
            resumed = main(['--headless', '--device', 'cpu', '--resume', str(checkpoint),
                            '--generations', '1', '--output', str(root / 'resumed')])
            with np.load(resumed / 'best.npz') as artifact:
                self.assertEqual(float(artifact['fitness']), 999999)
            rows = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['generation'], 2)
            early = main(['--headless', '--device', 'cpu', '--config', str(configuration),
                          '--generations', '3', '--stop-at-target', '--output', str(root / 'early')])
            self.assertEqual(len((early / 'metrics.jsonl').read_text().splitlines()), 1)

    def test_requested_gpu_failure_is_reported_instead_of_silently_using_cpu(self):
        from training import main
        with patch('jax.devices', side_effect=RuntimeError('No GPU')), \
             patch('sys.stderr', new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as error:
            main(['--headless', '--device', 'gpu'])
        self.assertEqual(error.exception.code, 2)
        self.assertIn('gpu device unavailable', stderr.getvalue())

    @unittest.skipUnless(importlib.util.find_spec('pygame'), 'requires pygame')
    def test_preview_uses_jax_game_and_closes_the_display(self):
        import os
        from rendering import TensorRenderer
        population = self.backend.algorithm.ask(self.initial)
        best = {'nodes': population[0][0], 'conns': population[1][0], 'fitness': -10}
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
                                     'PYGAME_HIDE_SUPPORT_PROMPT': '1'}):
            renderer = TensorRenderer(self.backend.algorithm, fps=0)
            try:
                renderer.show(self.initial, best, replace(self.config, max_steps=2), 0)
            finally:
                renderer.close()
            self.assertFalse(renderer.pygame.display.get_init())
