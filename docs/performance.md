# TensorNEAT CPU check and historical benchmark

## TensorNEAT/JAX migration

Tested locally on CPU with Python 3.12.15, JAX 0.11.2 and TensorNEAT revision
`ee204900e8e55ffe0ed756651ea871c70230dd34`. Population 256, 5 episodes,
1000 maximum steps, 16 node / 48 connection capacity, 3 generations, seed 42.

Generation compilation took 8.50 seconds; import, device initialization and
population setup are outside that number. The generation times below include
evaluation and evolution, with device synchronization before timing ends.
Artifact writing and rendering are outside these generation times.

| Generation | Simulated steps | Generation time | Steps/s | Mean fitness |
| --- | ---: | ---: | ---: | ---: |
| 0 | 33757 | 3.960 s | 8524 | 68.20 |
| 1 | 46440 | 3.299 s | 14075 | 155.70 |
| 2 | 57074 | 3.576 s | 15961 | 219.38 |

This is a functional CPU check, not a GPU benchmark or a controlled comparison
with the previous backend. Evolution, random streams and workload differ.
The TensorNEAT backend is slower on this CPU workload than the old process
pool. Its purpose is compiled, batched device execution; RTX 5090 performance
has not been measured. Completed lanes also wait for longer episodes within a
batch, so batching and population size need measurement on the target GPU.

```bash
python Snake_Game.py --headless --device cpu --population 256 --episodes 5 --max-steps 1000 --generations 3
```

## Historical neat-python process pool (commit 17634e6)

The following numbers and command belong to the previous backend. `--workers`
is no longer an option in the TensorNEAT version.

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

The historical backend ran CPU simulation and NEAT inference. No GPU
performance claims follow from these results.
