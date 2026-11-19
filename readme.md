# Snake AI

Gemeinsame, headless Snake-Simulation, auf der fünf Verfahren trainiert und
verglichen werden: **NEAT, REINFORCE, A2C, A3C und PPO**. Alle sehen dieselbe
Beobachtung, wählen aus denselben relativen Aktionen (links, geradeaus, rechts)
und werden auf denselben festen Bewertungsspielen gemessen. Planung und
Entscheidungen liegen im Seminar-Vault (`../Presentation/`).

Status: Proof of Concept im Umbau (Branch `feature/shared-simulation`). Alle
fünf Trainer und die Demo laufen; Replays fehlen noch.

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
uv run snake-ai train ppo                  # ppo | a2c | a3c | reinforce | neat
uv run snake-ai evaluate runs/<verfahren>/<lauf>/snapshots/<stand>
uv run snake-ai watch runs/<verfahren>/<lauf>/snapshots/<stand>
uv run snake-ai demo                       # alle fünf Verfahren nebeneinander
```

### Demo

`demo` zeigt NEAT, REINFORCE, A2C, A3C und PPO nebeneinander. Alle Felder
spielen dasselbe Spiel; die Spiele sind die festen Bewertungsspiele aus
`configs/evaluation.toml`, damit das Bild zu den Bewertungszahlen passt. Sind
alle Felder fertig, geht es nach zwei Sekunden mit dem nächsten Spiel weiter.

Pro Verfahren wird der neueste Lauf genommen, dessen Snapshots zur aktuellen
Spielkonfiguration und Beobachtungsversion passen. Ältere oder inkompatible
Läufe werden mit Hinweis übersprungen; fehlt ein Verfahren, bleibt sein Feld
leer und ist beschriftet.

| Taste | Wirkung |
| --- | --- |
| ←/→ | früherer/späterer Snapshot im markierten Feld; das Spiel startet dort neu |
| 1–5, Tab | Feld markieren |
| N/P | nächstes/vorheriges Bewertungsspiel für alle Felder |
| R | aktuelles Spiel neu starten |
| Leertaste | Pause |
| +/− | Geschwindigkeit verdoppeln/halbieren |
| Esc | beenden |

Optionen: `--start first|last|best` (Startstand pro Feld; `best` = höchster
Bewertungsmittelwert), `--run ppo=runs/ppo/<lauf>` (bestimmten Lauf nehmen,
wiederholbar), `--methods neat,ppo` (Auswahl und Reihenfolge), `--game 3`,
`--fps`, `--cell-size`.

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
| `configs/a3c.toml` | A3C (PyTorch-Multiprocessing) |

Die Anzahl der Worker und Threads ist pro Verfahren einstellbar. Vorgabe ist
höchstens 80 % CPU-Last auf dem Trainingsrechner.

## Aufbau

```
src/snake_ai/
  core/        Regeln, Zustand, Beobachtung; keine Grafik- oder ML-Importe
  envs/        Gymnasium-Adapter
  trainers/    sb3 (PPO, A2C), reinforce, neat_trainer, a3c, networks
  render/      pygame-Ansicht, liest nur den Zustand
  demo.py      Demo-Modell: Snapshot-Auswahl, gemeinsame Spiele (ohne pygame)
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
