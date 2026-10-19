import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import pickle
import tempfile
import unittest
from unittest.mock import patch

from evaluation import PopulationEvaluator
from training import main, restore_checkpoint, save_checkpoint


@unittest.skipUnless(importlib.util.find_spec("neat"), "requires neat-python")
class ParallelIntegrationTests(unittest.TestCase):
    def configuration(self, size=7):
        import neat
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             str(Path(__file__).resolve().parents[1] / 'config-feedforward.txt'))
        config.pop_size = size
        config.no_fitness_termination = True
        return config

    def test_serial_and_spawn_workers_produce_identical_results(self):
        import neat
        random.seed(14)
        config = self.configuration()
        population = neat.Population(config)
        genomes = list(population.population.items())
        with PopulationEvaluator(config, (11, 29, 37), 200) as serial:
            serial.evaluate(genomes, config)
        expected = copy.deepcopy(serial.results)
        with PopulationEvaluator(config, (11, 29, 37), 200, workers=2) as parallel:
            pool = parallel.pool
            for _ in range(2):
                parallel.evaluate(genomes, config)
                self.assertEqual(parallel.results, expected)
                self.assertIs(parallel.pool, pool)

    def test_checkpoint_resume_matches_uninterrupted_training(self):
        import neat

        def fresh():
            random.seed(19)
            return neat.Population(self.configuration())

        settings = {'seed': 19, 'seeds': (11, 29), 'max_steps': 100}
        uninterrupted = fresh()
        with PopulationEvaluator(uninterrupted.config, settings['seeds'], 100) as evaluator:
            uninterrupted.run(evaluator.evaluate, 2)
        expected = pickle_signatures(uninterrupted)
        interrupted = fresh()
        class LocalReporter(neat.reporting.BaseReporter):
            pass
        interrupted.add_reporter(LocalReporter())
        with PopulationEvaluator(interrupted.config, settings['seeds'], 100) as evaluator:
            interrupted.run(evaluator.evaluate, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pkl.gz'
            save_checkpoint(path, interrupted, settings, copy.deepcopy(interrupted.best_genome))
            resumed, restored_settings, best = restore_checkpoint(path)
            self.assertIs(resumed.species.reporters, resumed.reporters)
            self.assertEqual(restored_settings, settings)
            self.assertIsNotNone(best)
            with PopulationEvaluator(resumed.config, settings['seeds'], 100) as evaluator:
                resumed.run(evaluator.evaluate, 1)
        self.assertEqual(pickle_signatures(resumed), expected)

    def test_cli_runs_full_budget_and_restores_best_without_improvement(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config_path = root / 'config.txt'
            content = (Path(__file__).resolve().parents[1] / 'config-feedforward.txt').read_text()
            config_path.write_text(content.replace('fitness_threshold     = 10000', 'fitness_threshold     = -10'))
            first = main(['--headless', '--workers', '1', '--population', '7', '--episodes', '1',
                          '--max-steps', '1', '--generations', '3', '--config', str(config_path),
                          '--output', str(root / 'first')])
            rows = [json.loads(line) for line in (first / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['generation'] for row in rows], [0, 1, 2])
            checkpoint = first / 'checkpoint.pkl.gz'
            population, settings, best = restore_checkpoint(checkpoint)
            best.fitness = 999999  # Force a resume without an improvement.
            save_checkpoint(checkpoint, population, settings, best)
            resumed = main(['--headless', '--workers', '1', '--resume', str(checkpoint),
                            '--generations', '1', '--output', str(root / 'resumed')])
            with (resumed / 'best.pkl').open('rb') as handle:
                self.assertEqual(pickle.load(handle)['genome'].fitness, 999999)
            rows = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['generation'], 3)

    def test_cli_early_stop_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config_path = root / 'config.txt'
            content = (Path(__file__).resolve().parents[1] / 'config-feedforward.txt').read_text()
            config_path.write_text(content.replace('fitness_threshold     = 10000', 'fitness_threshold     = -10'))
            output = main(['--headless', '--workers', '1', '--population', '7', '--episodes', '1',
                           '--max-steps', '1', '--generations', '3', '--stop-at-target',
                           '--config', str(config_path), '--output', str(root / 'early')])
            self.assertEqual(len((output / 'metrics.jsonl').read_text().splitlines()), 1)

    @unittest.skipUnless(importlib.util.find_spec("pygame"), "requires pygame")
    def test_visual_and_headless_results_match_with_more_than_24_genomes(self):
        import neat
        from rendering import PopulationRenderer

        random.seed(5)
        config = self.configuration(25)
        genomes = list(neat.Population(config).population.items())
        with PopulationEvaluator(config, (7, 8), 100) as serial:
            serial.evaluate(genomes, config)
        with patch.dict(os.environ, {'SDL_VIDEODRIVER': 'dummy', 'SDL_AUDIODRIVER': 'dummy',
                                     'PYGAME_HIDE_SUPPORT_PROMPT': '1'}):
            renderer = PopulationRenderer(fps=0)
            with PopulationEvaluator(config, (7, 8), 100, renderer=renderer) as visual:
                visual.evaluate(genomes, config)
        self.assertEqual(visual.results, serial.results)


def pickle_signatures(population):
    return [(key, genome.fitness,
             sorted((node.key, node.bias, node.response, node.activation, node.aggregation)
                    for node in genome.nodes.values()),
             sorted((connection.key, connection.weight, connection.enabled)
                    for connection in genome.connections.values()))
            for key, genome in sorted(population.population.items())]
