"""NEAT coordinator, command line, metrics, and resumable run artifacts."""

import argparse
import copy
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import pickle
import random
import uuid

from evaluation import PopulationEvaluator


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def parser():
    cli = argparse.ArgumentParser(description="Train Snake with optional CPU parallelism")
    cli.add_argument("--headless", action="store_true", help="no pygame import or window")
    cli.add_argument("--workers", type=nonnegative, help="headless processes; 0/omitted: automatic")
    cli.add_argument("--generations", type=positive, default=1000, help="generations to run, including after resume")
    cli.add_argument("--population", type=positive, help="override configuration population size")
    cli.add_argument("--episodes", type=positive, help="shared evaluation seeds per genome (default: 5)")
    cli.add_argument("--seed", type=int, help="reproducible run seed (default: 42)")
    cli.add_argument("--max-steps", type=positive, help="hard episode step limit (default: 10000)")
    cli.add_argument("--fps", type=nonnegative, default=60, help="visual mode only; 0: unlimited")
    cli.add_argument("--stop-at-target", action="store_true", help="stop early at configured fitness threshold")
    cli.add_argument("--checkpoint-every", type=nonnegative, default=10, help="checkpoint interval; 0: final only")
    cli.add_argument("--config", type=Path, default=Path(__file__).with_name("config-feedforward.txt"))
    cli.add_argument("--output", type=Path, help="artifact directory (default: new directory under runs/)")
    cli.add_argument("--resume", type=Path, help="resume a checkpoint created by this trainer")
    return cli


def atomic_pickle(path, payload, compressed=False):
    temporary = path.with_name(path.name + ".tmp")
    opener = gzip.open if compressed else open
    with opener(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


@dataclass
class SerializableCounter:
    """Integer counter adapter for runtimes that cannot pickle itertools.count."""

    value: int

    def __iter__(self):
        return self

    def __next__(self):
        value = self.value
        self.value += 1
        return value


def prepare_checkpoint(population):
    for owner, name in ((population.config.genome_config, "node_indexer"),
                        (population.reproduction, "genome_indexer"),
                        (population.species, "indexer")):
        counter = getattr(owner, name)
        if counter is not None and not isinstance(counter, SerializableCounter):
            # Preserve the next value: consume from the discarded iterator and
            # start its replacement at that exact value, without skipping IDs.
            setattr(owner, name, SerializableCounter(next(counter)))


def save_checkpoint(path, population, settings, best, generation=None):
    prepare_checkpoint(population)
    species = copy.copy(population.species)
    # Reporters belong to the live coordinator, never to portable state.
    species.reporters = None
    atomic_pickle(path, {
        "version": 1,
        "config": population.config,
        "population": population.population,
        "species": species,
        "generation": population.generation if generation is None else generation,
        "genome_indexer": population.reproduction.genome_indexer,
        "ancestors": population.reproduction.ancestors,
        "random_state": random.getstate(),
        "settings": settings,
        "best": best,
    }, compressed=True)


def restore_checkpoint(path):
    import neat

    with gzip.open(path, "rb") as handle:
        saved = pickle.load(handle)
    if saved.get("version") != 1:
        raise ValueError("Unsupported checkpoint version")
    population = neat.Population(saved["config"],
                                 (saved["population"], saved["species"], saved["generation"]))
    population.species.reporters = population.reporters
    # NEAT 0.92 otherwise restarts its genome counter at 1 on resume.
    population.reproduction.genome_indexer = saved["genome_indexer"]
    population.reproduction.ancestors = saved["ancestors"]
    population.best_genome = saved["best"]
    random.setstate(saved["random_state"])
    return population, saved["settings"], saved["best"]


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    import neat

    if not args.headless and args.workers not in (None, 1):
        cli.error("multiple/automatic workers require --headless")
    if args.resume:
        if any(value is not None for value in (args.seed, args.population, args.episodes, args.max_steps)):
            cli.error("resume restores seed, population, episodes, and max-steps from the checkpoint")
        population, settings, best = restore_checkpoint(args.resume)
        config = population.config
    else:
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation, str(args.config.resolve()))
        if args.population is not None:
            config.pop_size = args.population
        if config.pop_size < 2:
            cli.error("population must be at least 2")
        run_seed = 42 if args.seed is None else args.seed
        seed_rng = random.Random(run_seed)
        settings = {"seed": run_seed, "seeds": tuple(seed_rng.randrange(2**63) for _ in range(args.episodes or 5)),
                    "max_steps": args.max_steps or 10000}
        random.seed(run_seed)
        population = neat.Population(config)
        best = None
    if config.genome_config.num_inputs != 5 or config.genome_config.num_outputs != 1:
        cli.error("Snake requires 5 inputs and 1 output")
    config.no_fitness_termination = not args.stop_at_target
    available = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    workers = 1 if not args.headless else min(args.workers or max(1, available - 1), config.pop_size)
    output = args.output or Path("runs") / uuid.uuid4().hex[:8]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists() and not args.resume:
        cli.error("output already contains a run; choose a new --output or use --resume")
    metadata = {**settings, "workers": workers, "population": config.pop_size,
                "generations_requested": args.generations, "headless": args.headless,
                "stop_at_target": args.stop_at_target, "resume": str(args.resume) if args.resume else None}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    renderer = None
    if not args.headless:
        from rendering import PopulationRenderer
        renderer = PopulationRenderer(args.fps)
        renderer.generation = population.generation

    evaluator = PopulationEvaluator(config, settings["seeds"], settings["max_steps"], workers, renderer)

    class RunReporter(neat.reporting.BaseReporter):
        def __init__(self):
            self.best = copy.deepcopy(best)
            if self.best is not None:
                atomic_pickle(output / "best.pkl", {"genome": self.best, "config": config, "settings": settings})

        def post_evaluate(self, config, genomes, species, generation_best):
            prepare_checkpoint(population)
            if self.best is None or generation_best.fitness > self.best.fitness:
                self.best = copy.deepcopy(generation_best)
                atomic_pickle(output / "best.pkl", {"genome": self.best, "config": config, "settings": settings})
            results = evaluator.results
            episodes = [episode for result in results.values() for episode in result.episodes]
            steps = sum(episode.steps for episode in episodes)
            metrics = {"generation": population.generation, "genomes": len(results),
                       "episodes": len(episodes), "steps": steps, "seconds": evaluator.seconds,
                       "steps_per_second": steps / max(evaluator.seconds, 1e-9),
                       "best_fitness": generation_best.fitness,
                       "mean_fitness": sum(result.fitness for result in results.values()) / len(results),
                       "mean_score": sum(episode.score for episode in episodes) / len(episodes),
                       "max_score": max(episode.score for episode in episodes)}
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            print(f"Generation {population.generation}: best={metrics['best_fitness']:.1f} "
                  f"mean={metrics['mean_fitness']:.1f} score={metrics['mean_score']:.2f} "
                  f"steps/s={metrics['steps_per_second']:.0f}", flush=True)

        def end_generation(self, config, genomes, species):
            next_generation = population.generation + 1
            if args.checkpoint_every and next_generation % args.checkpoint_every == 0:
                save_checkpoint(output / "checkpoint.pkl.gz", population, settings, self.best, next_generation)

    reporter = RunReporter()
    population.add_reporter(reporter)
    print(f"Mode: {'headless' if args.headless else 'visual'}, workers={workers}, "
          f"population={config.pop_size}, episodes={len(settings['seeds'])}, output={output}", flush=True)
    try:
        with evaluator:
            population.run(evaluator.evaluate, args.generations)
    except KeyboardInterrupt:
        print("Interrupted; saving the current generation for reevaluation.")
    finally:
        save_checkpoint(output / "checkpoint.pkl.gz", population, settings, reporter.best)
    print(f"Saved artifacts to {output}")
    return output
