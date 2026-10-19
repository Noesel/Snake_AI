# CPU benchmark

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
