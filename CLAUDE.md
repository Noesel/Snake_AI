# CLAUDE.md — Snake AI

Instructions for coding agents in this repository. **Goal: a clean, object-oriented proof of concept in modern, idiomatic Python, with minimum necessary complexity.** Much of the work is exploratory (training experiments, observation encodings, hyperparameters); code should make experiments easy to run and compare, not be production-hardened.

## Project context

- A shared, headless Snake simulation used by five training methods: **NEAT, REINFORCE, A2C, A3C and PPO**. Plus a demo that shows five independent games side by side with selectable snapshots, and replay of selected games. Built for a seminar talk (AI in Games).
- The codebase is about **ten years old**. Existing structure, libraries and versions are not a reason to keep them; see "Dependencies".
- Planning lives in the vault next to this repo (`../Presentation/`):
  - [Implementierungsplan](../Presentation/Implementierungsplan.md): steps and scope.
  - [Entscheidungslog](../Presentation/Entscheidungslog.md): decisions on rules, rewards, observation and architecture. Agreed game rules come from there, not from the old code.
  - [Zielrechner](../Presentation/Zielrechner.md): target hardware (i9-14900KF, 32 logical CPUs, RTX 5090, Python 3.14).
- Record relevant decisions and changes in the Entscheidungslog as new entries (decided vs. proposed vs. open). Do not overwrite earlier entries.

## Scope

- Implement only the current, commit-sized step of the plan. Do not add components, abstractions or pipeline stages the plan does not call for; if something seems missing, say so.
- Planned stack (versions to be confirmed in plan step 2): PyTorch for REINFORCE and A3C, Stable-Baselines3 for A2C and PPO, Gymnasium as the RL adapter, a NEAT library for NEAT, pygame-ce (or similar) for rendering only.
- Avoid speculative infrastructure (databases, web servers, experiment-tracking platforms, distributed frameworks, GPU-vectorized simulation) unless measurements or a documented decision call for it.

## Architecture

- **Simulation core** owns state and rules, with `reset(seed)` and `step(action)`. Relative actions: left / straight / right. It has no imports from pygame, PyTorch, Gymnasium or NEAT libraries, and it can be stepped headless.
- **Adapters** wrap the core per consumer: Gymnasium env for RL, NEAT evaluation, model playback. One observation format, versioned, shared by all methods.
- **Rendering** reads state and never changes rules. Training never needs a display.
- **Trainers** depend on the core through adapters; each method's hyperparameters live in its own config, shared settings in a common config. Every run stores its effective config, seeds and library versions next to its snapshots.
- A composition root (CLI entry point) wires config, core, adapter and trainer through constructor injection. No global mutable state, no module-level side effects such as `pygame.init()` on import.

## Design principles

1. **OOP first, but pragmatic:** model cohesive concepts (game state, snake, board, trainer, snapshot) as classes. Use pure functions for stateless transformations; do not create classes just to wrap functions.
2. **SOLID where it pays off:** single responsibility, small interfaces, composition over inheritance. Use a typed `Protocol` only at real boundaries (e.g. trainer or policy contracts), not one per class.
3. **High cohesion, low coupling:** each module owns one responsibility. No circular imports, hidden side effects or catch-all `utils.py` / `manager.py` modules.
4. **KISS / YAGNI / DRY:** explicit code and the smallest useful abstraction. Do not generalize for hypothetical future methods or games.
5. **Correctness of the simulation first:** rules must match the Entscheidungslog; no silent fallbacks or undocumented behavior changes. Reproducibility through explicit seeds and RNG objects instead of the global `random` module.
6. **Immutable value objects** (`@dataclass(frozen=True)`) for configs, observations and step results where practical.

## Clean Python

- Follow **PEP 8**, **PEP 257** and modern typing (`X | None`, built-in generics). Honor the Python version declared in `pyproject.toml`.
- `snake_case` for modules, functions and variables; `PascalCase` for classes; `UPPER_SNAKE_CASE` for constants. New modules do not use the legacy `c_` prefix.
- Annotate public parameters and return values; avoid unqualified `Any`, unnecessary `# type: ignore`, wildcard imports and mutable default arguments.
- `ruff format` and `ruff check` are formatter and linter, configured in `pyproject.toml`. Do not invent unconfigured commands.
- Small, cohesive methods with early returns and clear names. Comment the non-obvious **why**; short docstrings for public contracts.
- `logging` instead of `print` in library code (CLI output and progress may print), `pathlib` for paths, context managers for resources, explicit narrow exceptions. Never `except Exception: pass`.
- No magic numbers scattered in code: board size, energy, rewards and hyperparameters come from typed configuration.

## Dependencies

- **Question every existing dependency and version.** Old pins (e.g. `pygame==2.0.1`, `neat-python==0.92`) are not binding. For each library, check whether it is still maintained, works on the target Python and hardware, and is the right tool. Prefer the current stable release unless there is a concrete reason not to, and record the choice in the Entscheidungslog.
- Metadata and direct dependencies in **`pyproject.toml`**, managed with **uv** and a committed `uv.lock`. Separate optional groups where useful (e.g. `dev`, `render`). One lock mechanism only; retire `Requirements.txt` once the move is done.
- Declare every directly imported package; remove unused ones. Prefer the standard library where it is enough.
- Before adding a package, check need, maintenance, license and transitive weight; justify it in the change summary.
- Do not commit `.venv/`, run outputs, snapshots or large artifacts unless explicitly decided.

## Performance

- Performance numbers and optimization decisions count only when measured on the target desktop (see Zielrechner). Measure before and after; no blanket optimization.
- Respect the **80 % load limit**: worker processes, PyTorch threads and A3C workers together stay within the configured budget (proposed: 24–25 of 32 logical CPUs). Make worker and thread counts configurable.
- Small networks do not automatically benefit from CUDA; compare CPU and GPU before choosing. Use the `spawn` start method when CUDA is used in subprocesses.

## Tests

- Tests are **allowed, not mandatory**. Write them only where they genuinely help the project, for example:
  - simulation rules and edge cases that are easy to break (collision, tail cell, food on the last energy step, full board),
  - seed reproducibility and replay determinism,
  - snapshot save and load round-trips.
- No tests for trivial code, rendering details or exploratory training scripts. Do not keep tests that only pin down legacy implementation details; adapt or delete them when the design changes.
- A short smoke run (a few training steps, save, load, play back) is often more useful than unit tests for trainer integration.
- Run tests with `uv run pytest` once configured. Do not claim tests or runs passed unless you ran them.

## Agent workflow

- One coherent change at a time; no unrelated refactors or dependency changes mixed in.
- **Never commit or push unless explicitly requested.**
- After changes, run what is configured: `uv run ruff format --check .`, `uv run ruff check .` and, where tests exist for the touched area, `uv run pytest`.
- Review the diff for unintended files, cross-layer imports (e.g. pygame in the core) and rule changes not backed by the Entscheidungslog.
- Report: what changed, what was actually run, remaining limitations, and one suggested conventional commit message.

### Definition of Done

- [ ] Meets the current plan step without scope creep.
- [ ] Core stays independent of rendering and training libraries; clear dependency direction, no cycles.
- [ ] Typed, formatted, lint-clean code.
- [ ] Rules match the Entscheidungslog; runs are reproducible from config and seed.
- [ ] Dependencies questioned and documented; no secrets or large artifacts committed.

## References

- [PEP 8](https://peps.python.org/pep-0008/) · [PEP 257](https://peps.python.org/pep-0257/)
- [Writing pyproject.toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [Gymnasium environment creation](https://gymnasium.farama.org/introduction/create_custom_env/)
