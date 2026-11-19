# CPU benchmark

## All five trainers (2026-11-19)

Default configs, one run each (seed 0), target desktop, CPU only. Evaluation:
mean score on the 20 fixed games of `configs/evaluation.toml`. Single runs,
no statistical claims.

| Method | Budget | Parallelism | Wall time | Throughput | Final eval mean (max) |
| --- | --- | --- | ---: | ---: | ---: |
| PPO (SB3) | 3 M env steps | 8 envs, 4 torch threads | 139 s | ~21 600 steps/s | 8.50 (24) |
| A2C (SB3) | 3 M env steps | 16 envs, 4 torch threads | 86 s | ~35 000 steps/s | 9.65 (21) |
| REINFORCE | 3 M env steps | 16 episodes in lockstep, 4 threads | 138 s | ~21 700 steps/s | 6.05 (11) |
| A3C | 3 M env steps | 16 worker processes, 1 thread each | 103 s | ~29 600 steps/s | 3.55 (9), unstable |
| NEAT | 300 generations x 256 genomes x 5 games | 24 worker processes | 132 s | ~0.44 s/generation | 6.00 (12) |

## Shared simulation core (2026-11-19)

`snake_ai.core.SnakeGame` on the target desktop, Python 3.14.10, numpy 2.5,
10 x 10 board, uniformly random actions (short episodes, resets included),
single process, best of three runs of 300 000 steps:

| Work per step | Steps/s |
| --- | ---: |
| `step()` only | 805 791 |
| `step()` + `ObservationEncoder.encode()` (107 values) | 307 810 |

Encoding costs more than the rules. For comparison, SB3 PPO with a small MLP
reached about 3 300 steps/s on CartPole on this machine, so the simulation is
not the bottleneck for RL training. No optimization planned until a trainer
measurement says otherwise.

## PPO training throughput (2026-11-19)

`snake-ai train ppo` with `configs/ppo.toml` (8 environments in one process,
4 torch threads, CPU, MLP policy), 3 000 000 steps including 13 snapshot
evaluations: 139 s wall time, about 21 600 environment steps/s.

## Target desktop baseline (2026-11-19)

Measured on the target desktop: Intel Core i9-14900KF (24 cores / 32 logical
CPUs), CachyOS, Python 3.14.10, neat-python 0.92, commit 39a742c. Headless,
8 episodes per genome, 8 generations, 2000 steps maximum, seed 42, final
checkpoint only. Median of repeated runs (3 for population 256, 2 for 2048).
"CPU busy" is whole-system utilization over the full run, including process
startup. Desktop applications were running.

Population 256:

| Workers | Total runtime | Evaluation steps/s after first generation | CPU busy |
| --- | ---: | ---: | ---: |
| 1 | 2.44 s | 350384 | 4 % |
| 2 | 1.39 s | 661504 | 7 % |
| 4 | 0.85 s | 1229712 | 14 % |
| 8 | 0.57 s | 2163794 | 21 % |
| 12 | 0.59 s | 2245147 | 28 % |
| 16 | 0.60 s | 2125197 | 38 % |
| 20 | 0.55 s | 2504220 | 44 % |
| 24 | 0.51 s | 2791755 | 50 % |

Population 2048:

| Workers | Total runtime | Evaluation steps/s after first generation | CPU busy |
| --- | ---: | ---: | ---: |
| 1 | 17.53 s | 353317 | 4 % |
| 4 | 5.10 s | 1314585 | 13 % |
| 8 | 2.99 s | 2451830 | 23 % |
| 12 | 2.67 s | 2868299 | 32 % |
| 16 | 2.41 s | 3243715 | 40 % |
| 20 | 2.29 s | 3526981 | 48 % |
| 24 | 2.22 s | 3672798 | 56 % |

Fitness results were identical for every worker count and repetition.

Findings:

- Evaluation throughput scales to about 10x at 24 workers, but total runtime
  levels off from about 16 workers. Above 8 workers the scaling is clearly
  sublinear (hybrid CPU with efficiency cores, coordination overhead).
- Profiling the 24-worker run (population 2048) shows that the main process
  spends its time on per-genome inter-process communication
  (`pool.map(..., chunksize=1)`) and on serial NEAT reproduction and
  speciation (about a quarter of the profiled runtime).
- A larger chunksize (`len(items) // (workers * 4)`), tested in a throwaway
  copy, reduced the 24-worker runtime only from 2.22 s to 2.10 s. It is not the
  main lever.
- The 80 % load limit was never reached; with 24 workers, utilization stayed at
  about 50–56 %.

## Historical measurement (other machine)

Measured locally with Python 3.14.8, NEAT 0.92 and 12 available logical CPUs.
This is not a measurement on the user's gaming PC or RTX 5090.

Settings: headless, population 256, 8 episodes per genome, 8 generations,
2000 steps maximum per episode, seed 42, final checkpoint only.

| Workers | Total runtime | Evaluation steps/s after first generation |
| --- | ---: | ---: |
| 1 | 6.202 s | 135824 |
| 2 | 3.942 s | 228932 |
| 4 | 2.665 s | 364790 |
| 8 | 2.518 s | 400903 |

All four runs produced identical generation-by-generation population sizes,
step counts, best fitnesses and mean fitnesses. Runtime includes coordination,
evolution, process startup and artifact writing. Evaluation throughput excludes
the first generation to remove worker startup and sums steps/time across the
remaining generations. These are single short measurements, not confidence
intervals or guaranteed speedups.

In this workload, 8 workers reduced total runtime by about 2.46x compared with
one worker. Gains from 4 to 8 workers were small. Use the target PC's own
measurements to choose worker count; more workers are not always faster.

Example command (change worker count and use a new output directory each time):

```bash
python Snake_Game.py --headless --workers 4 --population 256 --episodes 8 --generations 8 --max-steps 2000 --seed 42 --checkpoint-every 0
```

The backend runs CPU simulation and NEAT inference. No GPU performance claims
follow from these results. GPU batching and GPU-resident simulation are future
work.
