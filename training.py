"""TensorNEAT/JAX training, device selection, metrics, and portable run state."""

import argparse
from dataclasses import asdict, replace
import gzip
import json
import os
from pathlib import Path
import pickle
import time
import uuid

from configuration import TrainingConfig, TENSORNEAT_REVISION, load_config


FORMAT = 'snake-tensorneat-v1'


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return value


def parser():
    cli = argparse.ArgumentParser(description='Train Snake using TensorNEAT and batched JAX simulation')
    cli.add_argument('--headless', action='store_true', help='no pygame import or preview window')
    cli.add_argument('--device', choices=('auto', 'cpu', 'gpu'), default='auto')
    cli.add_argument('--generations', type=positive, default=1000, help='additional generations to run')
    for name in ('population', 'episodes', 'max-steps', 'max-nodes', 'max-conns', 'batch-size'):
        cli.add_argument('--' + name, type=positive)
    cli.add_argument('--seed', type=nonnegative)
    cli.add_argument('--fps', type=nonnegative, default=60, help='preview FPS; 0: unlimited')
    cli.add_argument('--stop-at-target', action='store_true', help='stop early at configured mean fitness target')
    cli.add_argument('--checkpoint-every', type=nonnegative, default=10, help='0: final checkpoint only')
    cli.add_argument('--config', type=Path, default=Path(__file__).with_name('config-tensorneat.json'))
    cli.add_argument('--output', type=Path, help='artifact directory; default: new directory under runs/')
    cli.add_argument('--resume', type=Path, help='TensorNEAT checkpoint created by this trainer')
    return cli


def atomic_pickle(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    with gzip.open(temporary, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def save_best(path, config, best):
    import numpy as np
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, nodes=best['nodes'], conns=best['conns'],
                            fitness=best['fitness'], generation=best['generation'],
                            config=json.dumps(asdict(config)), revision=TENSORNEAT_REVISION)
    temporary.replace(path)


def save_checkpoint(path, config, state, best):
    import jax
    atomic_pickle(path, {'format': FORMAT, 'revision': TENSORNEAT_REVISION,
                         'jax_version': jax.__version__, 'config': asdict(config),
                         'state': jax.device_get(state), 'best': best})


def restore_checkpoint(path):
    import jax
    import jax.numpy as jnp
    try:
        with gzip.open(path, 'rb') as handle:
            saved = pickle.load(handle)
    except ModuleNotFoundError as error:
        raise ValueError('Legacy neat-python checkpoints cannot be resumed with TensorNEAT') from error
    if saved.get('format') != FORMAT:
        raise ValueError('This is not a TensorNEAT Snake checkpoint; neat-python state is incompatible')
    if saved.get('revision') != TENSORNEAT_REVISION or saved.get('jax_version') != jax.__version__:
        raise ValueError('Resume requires the same TensorNEAT revision and JAX version')
    config = TrainingConfig(**saved['config']).validate()
    return config, jax.tree.map(jnp.asarray, saved['state']), saved['best']


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    overrides = {name: getattr(args, name) for name in
                 ('population', 'episodes', 'max_steps', 'max_nodes', 'max_conns', 'batch_size', 'seed')
                 if getattr(args, name) is not None}
    if args.resume and overrides:
        cli.error('resume restores population, episodes, seed, capacities and step/batch limits')
    # Respect user-supplied allocator settings; do not reserve most of a gaming GPU.
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    import numpy as np

    try:
        devices = jax.devices() if args.device == 'auto' else jax.devices(args.device)
    except RuntimeError as error:
        cli.error(f'{args.device} device unavailable: {error}. See Requirements-gpu.txt for CUDA installation.')
    device = devices[0]
    jax.config.update('jax_compilation_cache_dir', str(Path('runs/.jax-cache').resolve()))
    with jax.default_device(device):
        from tensor_backend import TensorBackend
        try:
            if args.resume:
                config, state, best = restore_checkpoint(args.resume)
                state = jax.device_put(state, device)
            else:
                config = replace(load_config(args.config), **overrides).validate()
                state, best = None, None
        except (ValueError, TypeError, OSError) as error:
            cli.error(str(error))
        output = args.output or Path('runs') / uuid.uuid4().hex[:8]
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'run.json').exists() and not args.resume:
            cli.error('output already contains a run; choose a new --output or use --resume')
        metadata = {'format': FORMAT, 'revision': TENSORNEAT_REVISION,
                    'jax_version': jax.__version__, 'config': asdict(config),
                    'device': str(device), 'platform': device.platform,
                    'generations_requested': args.generations, 'headless': args.headless,
                    'stop_at_target': args.stop_at_target,
                    'resume': str(args.resume) if args.resume else None}
        (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
        backend = TensorBackend(config)
        if state is None:
            state = backend.setup()
        if best is not None:
            save_best(output / 'best.npz', config, best)
        renderer = None
        print(f'TensorNEAT on {device}: population={config.population}, episodes={config.episodes}, '
              f'batch={config.batch_size or config.population}, output={output}', flush=True)
        try:
            print('Compiling evolution and batched simulation...', flush=True)
            started = time.perf_counter()
            compiled = backend.compile(state)
            compile_seconds = time.perf_counter() - started
            print(f'Compilation ready in {compile_seconds:.2f}s', flush=True)
            if not args.headless:
                from rendering import TensorRenderer
                renderer = TensorRenderer(backend.algorithm, fps=args.fps)
            for _ in range(args.generations):
                generation = int(state.generation)
                started = time.perf_counter()
                next_state, population, statistics = compiled(state)
                jax.block_until_ready((next_state, population, statistics))
                seconds = time.perf_counter() - started
                state = next_state
                fitness, scores, maxima, steps = jax.device_get(statistics)
                if not np.all(np.isfinite(fitness)):
                    raise RuntimeError('Non-finite population fitness')
                index = int(np.argmax(fitness))
                if best is None or float(fitness[index]) > best['fitness']:
                    nodes, conns = jax.device_get((population[0][index], population[1][index]))
                    best = {'nodes': nodes, 'conns': conns, 'fitness': float(fitness[index]),
                            'generation': generation}
                    save_best(output / 'best.npz', config, best)
                total_steps = int(np.sum(steps))
                metrics = {'generation': generation, 'genomes': config.population,
                           'episodes': config.population * config.episodes,
                           'steps': total_steps, 'seconds': seconds,
                           'steps_per_second': total_steps / max(seconds, 1e-9),
                           'best_fitness': float(np.max(fitness)), 'mean_fitness': float(np.mean(fitness)),
                           'mean_score': float(np.mean(scores)), 'max_score': int(np.max(maxima)),
                           'compile_seconds': compile_seconds if _ == 0 else 0.0}
                with (output / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(metrics) + '\n')
                print(f"Generation {generation}: best={metrics['best_fitness']:.1f} "
                      f"mean={metrics['mean_fitness']:.1f} score={metrics['mean_score']:.2f} "
                      f"steps/s={metrics['steps_per_second']:.0f}", flush=True)
                if args.checkpoint_every and int(state.generation) % args.checkpoint_every == 0:
                    save_checkpoint(output / 'checkpoint.pkl.gz', config, state, best)
                if renderer is not None:
                    renderer.show(state, best, config, generation)
                if args.stop_at_target and float(np.max(fitness)) >= config.fitness_target:
                    print('Configured fitness target reached.')
                    break
        except KeyboardInterrupt:
            print('Interrupted; saving the last completed generation.')
        finally:
            if renderer is not None:
                renderer.close()
            save_checkpoint(output / 'checkpoint.pkl.gz', config, state, best)
        print(f'Saved artifacts to {output}')
        return output
