# Snake AI

Gemeinsame, headless Snake-Simulation, auf der fünf Verfahren trainiert und
verglichen werden: **NEAT, REINFORCE, A2C, A3C und PPO**. Alle sehen dieselbe
Beobachtung, wählen aus denselben relativen Aktionen (links, geradeaus, rechts)
und werden auf denselben festen Bewertungsspielen gemessen. Planung und
Entscheidungen liegen im Seminar-Vault (`../Presentation/`).

Status: Proof of Concept im Umbau (Branch `feature/shared-simulation`). A3C
fehlt noch.

## Installation

Benötigt [uv](https://docs.astral.sh/uv/) und Python 3.14.

```bash
uv sync                    # Laufzeit + Entwicklungswerkzeuge
uv sync --extra render     # zusätzlich pygame-ce für Fenster
```

PyTorch kommt als CUDA-Build (etwa 5 GB). Die kleinen Netze trainieren auf
der CPU schneller; siehe `docs/performance.md`.

## Benutzung

```bash
uv run snake-ai play                       # selbst spielen: Pfeiltasten/A-D, R neu, Esc
uv run snake-ai train ppo                  # ppo | a2c | reinforce | neat
uv run snake-ai evaluate runs/<verfahren>/<lauf>/snapshots/<stand>
uv run snake-ai watch runs/<verfahren>/<lauf>/snapshots/<stand>
```

`train` läuft ohne Fenster. Jeder Lauf legt unter
`runs/<verfahren>/<zeit>_<id>/` ab:

- `config.json`: effektive Konfiguration (Spiel, Beobachtung, Training, Bewertung)
- `versions.json`: Python, Bibliotheken, Simulations- und Beobachtungsversion, Git-Commit
- `metrics.jsonl`: eine Zeile pro Snapshot mit Trainings- und Bewertungskennzahlen
- `snapshots/<einheit>_<fortschritt>/`: Modell und `meta.json`

Fortschritt zählt bei RL in Environment Steps und bei NEAT in Generationen.
Die Zahlen sind nicht direkt vergleichbar.

## Konfiguration

| Datei | Inhalt |
| --- | --- |
| `configs/game.toml` | Spielregeln: Feldgröße, Startlänge, Energie, Rewards |
| `configs/evaluation.toml` | feste Bewertungsspiele, für alle Verfahren gleich |
| `configs/ppo.toml`, `a2c.toml` | Stable-Baselines3-Training und Hyperparameter |
| `configs/reinforce.toml` | REINFORCE (PyTorch) |
| `configs/neat.toml`, `neat.ini` | NEAT-Lauf und neat-python-Einstellungen |

Die Anzahl der Worker und Threads ist pro Verfahren einstellbar. Vorgabe ist
höchstens 80 % CPU-Last auf dem Trainingsrechner.

## Aufbau

```
src/snake_ai/
  core/        Regeln, Zustand, Beobachtung; keine Grafik- oder ML-Importe
  envs/        Gymnasium-Adapter
  trainers/    sb3 (PPO, A2C), reinforce, neat_trainer, networks
  render/      pygame-Ansicht, liest nur den Zustand
  evaluation.py, runs.py, settings.py, policies.py, cli.py
```

## Entwicklung

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
```

Die Tests prüfen gezielt Spielregeln, das Beobachtungsformat und für jeden
Trainer die Kette Training → Snapshot → Laden → Bewertung.
