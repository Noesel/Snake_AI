# Snake AI

> **Umbau läuft** (Branch `feature/shared-simulation`): Ein gemeinsamer,
> headless Simulationskern für NEAT, REINFORCE, A2C, A3C und PPO entsteht unter
> `src/snake_ai/`. Der unten beschriebene NEAT-Code ist der alte Stand und wird
> schrittweise ersetzt.
>
> ```bash
> uv sync                    # Python 3.14, Abhängigkeiten aus pyproject.toml
> uv run pytest              # Regeltests des neuen Kerns
> uv run ruff check .
> ```
>
> Spielregeln: `configs/game.toml`. Planung und Entscheidungen liegen im
> Seminar-Vault (`../Presentation/`).

Snake mit NEAT: fünf relative Eingaben, eine Entscheidung für Links/Geradeaus/Rechts.
Training ist mit Anzeige oder ohne Grafik auf mehreren CPU-Kernen möglich.

## Installation

Python 3 mit `pip` und `venv` wird benötigt. Unter Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Unter Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Für Training mit Anzeige:

```bash
python -m pip install -r Requirements.txt
```

Für einen reinen Trainingsrechner ohne Anzeige genügt:

```bash
python -m pip install -r Requirements-headless.txt
```

Getestete Versionen: Python 3.14.8, `neat-python==0.92`, optional
`pygame-ce==2.5.7`. Headless Training benötigt kein Pygame und keinen
SDL-Dummy-Treiber. Pygame CE wird für die Anzeige als `pygame` importiert.

## Start mit Anzeige

```bash
python Snake_Game.py --generations 100
```

Die Anzeige zeigt bis zu 24 Spiele gleichzeitig. Größere Populationen werden
in Gruppen angezeigt; die Populationsgröße ist nicht mehr an 24 gebunden.
Jedes Genom wird standardmäßig über fünf Episoden bewertet. Die Anzeige läuft
mit 60 FPS; `--fps 0` hebt die Begrenzung auf. Fenster schließen oder `Strg+C`
beendet den Lauf und speichert einen Checkpoint.

## Headless und parallel trainieren

Beispiel für einen größeren Lauf:

```bash
python Snake_Game.py --headless --workers 8 --population 256 --episodes 5 --generations 1000 --seed 42
```

Ein kurzer Funktionstest:

```bash
python Snake_Game.py --headless --workers 2 --population 32 --episodes 3 --generations 3
```

`--workers 1` wertet seriell aus. `--workers 0` oder ein weggelassenes
`--workers` wählt im Headless-Modus automatisch bis zu einen Prozess pro
verfügbarem CPU-Kern, abzüglich eines Kerns. Das ist eine Heuristik, keine
automatische Leistungsoptimierung. Die Zahl wird auf die konfigurierte
Populationsgröße begrenzt. Mehrere Worker erfordern `--headless`.

Die Worker bleiben über Generationen bestehen. Ein Auftrag umfasst ein Genom
mit allen Episoden; Konfiguration und Seeds werden nur beim Worker-Start
übertragen. Der Koordinator sammelt Ergebnisse und führt die Evolution aus.
Prozesse verwenden `spawn`, damit beim Start keine Fenster oder CUDA-Kontexte
vererbt werden. Getestet wurde unter Linux; ein Test auf dem Zielrechner mit
Windows steht noch aus.

Alle Genome und Generationen verwenden denselben Satz Episoden-Seeds.
Simulationen besitzen eigene Zufallszahlengeneratoren und verändern den
Zufallszustand der Evolution nicht. Bei gleicher Konfiguration, gleichem Seed
und derselben Python-/NEAT-Version stimmen serielle und parallele Ergebnisse
überein. Die Fitness ist der Mittelwert über die Episoden. Feste Trainings-Seeds
können überangepasst werden; eine separate Bewertung auf neuen Seeds ist ein
späterer Ausbau.

## Ergebnisse und Fortsetzen

Jeder Lauf legt ein neues Verzeichnis unter `runs/` an. Mit
`--output runs/mein-lauf` lässt sich ein eigenes Verzeichnis wählen.

| Datei | Inhalt |
| --- | --- |
| `best.pkl` | Bisher bestes Genom mit NEAT-Konfiguration und Auswertungseinstellungen |
| `checkpoint.pkl.gz` | Population, Arten, Zähler, Zufallszustand und Einstellungen zum Fortsetzen |
| `metrics.jsonl` | Pro Generation Fitness, Punkte, Schritte und Auswertungsdurchsatz |
| `run.json` | Einstellungen des gestarteten Laufs |

Das beste Genom wird bei Verbesserung gespeichert. Checkpoints werden alle
zehn Generationen und am Laufende gespeichert. `--checkpoint-every 1` speichert
nach jeder Generation; `--checkpoint-every 0` nur am Ende. Dateien werden über
eine temporäre Datei ersetzt. Bei `Strg+C` kann die zuletzt begonnene Generation
beim Fortsetzen erneut ausgewertet werden.

```bash
python Snake_Game.py --headless --workers 8 --resume runs/mein-lauf/checkpoint.pkl.gz --generations 1000
```

Das startet bis zu 1000 **weitere** Generationen. Ohne `--output` entsteht ein
neues Ergebnisverzeichnis. Seed, Populationsgröße, Episoden und Schrittlänge
werden aus dem Checkpoint übernommen. Worker-Zahl und Anzeigemodus lassen sich
ändern. Das Dateiformat gehört zu diesem Trainer, nicht zu NEATs eingebautem
Checkpointer; dieselbe Python-/NEAT-Version wird vorausgesetzt.

## Trainingsparameter

| Parameter | Einstellung und Wirkung |
| --- | --- |
| Ein-/Ausgänge | `5` Eingänge, `1` Ausgang mit `tanh` |
| Aktion | Ausgang über `0.5`: rechts, unter `-0.5`: links, sonst geradeaus |
| Population | Standard `24`; über `--population` frei einstellbar, mindestens `2` |
| Episoden | Standard `5`; Fitness wird gemittelt |
| Energie | Anfangs `20` Züge, pro Futter zusätzlich `20` |
| Fitness pro Episode | `+100` pro Futter, `-10` am Episodenende, auch beim Schrittlimit |
| Schrittlimit | Standard `10000` pro Episode, einstellbar mit `--max-steps` |
| Generationen | Standard `1000`; die angeforderte Zahl wird regulär durchlaufen |
| Früher Abbruch | Nur mit `--stop-at-target`; verwendet das Fitnessziel `10000` aus der Konfiguration |
| Stagnation | `max_stagnation = 1000` bleibt unverändert und sortiert innerhalb eines Laufs mit 1000 Generationen praktisch keine Arten aus |

Die Zählung im Terminal beginnt bei Generation `0`. Ein vollständig belegtes
Spielfeld beendet die Episode; die Futterplatzierung kann nicht endlos suchen.
Eine manuelle Tastatursteuerung ist nicht eingebaut.

## KI-Eingaben

| Reihenfolge | Information | Werte |
| --- | --- | --- |
| 1 | Gefahr vorne: Wand oder Körper beim nächsten Zug | `0` oder `1` |
| 2 | Gefahr nach Linksdrehung | `0` oder `1` |
| 3 | Gefahr nach Rechtsdrehung | `0` oder `1` |
| 4 | Futter vor/hinter der Schlange | `1`: vorne, `-1`: hinten, `0`: seitlich |
| 5 | Futter rechts/links der Schlange | `1`: rechts, `-1`: links, `0`: auf der Blickachse |

Die Entscheidung erfolgt vor jedem Zug, einschließlich des ersten. Die
Schwanzzelle gilt als frei, wenn sie im nächsten Zug verlassen wird. Neues
Futter wird nach dem Wachstum platziert und liegt nicht im Körper.

## Effizienz und RTX 5090

Dieses Backend nutzt **CPU-Prozesse**, kein CUDA. NEAT 0.92 wertet kleine Netze
mit veränderlicher Topologie in Python aus. Eine RTX 5090 beschleunigt diesen
Code daher nicht automatisch. Ein zukünftiges GPU-Backend müsste viele
Simulationen und Netzberechnungen gebündelt auf der GPU durchführen und häufige
Übertragungen pro Zug vermeiden. Die getrennte Auswertung bietet dafür einen
klaren Austauschpunkt; das Backend ist hier noch nicht implementiert.

Die festgelegte NEAT-Version `0.92` von 2017 bleibt hier die getestete
CPU-Vergleichsbasis. Aktuell auf [PyPI](https://pypi.org/project/neat-python/)
ist `2.0.0` vom März 2026. Ein Upgrade ist eine eigene Migration; alte
Checkpoints sind laut [Changelog](https://github.com/CodeReclaimers/neat-python/blob/master/CHANGELOG.md)
nicht mit Versionen ab 1.0 kompatibel. Für GPU-NEAT ist
[TensorNEAT](https://github.com/EMI-Group/tensorneat) mit JAX ein Kandidat;
[EvoTorch](https://github.com/nnaisense/evotorch) bietet andere
Evolutionsverfahren mit PyTorch. Keines davon wurde auf der Zielhardware
dieses Projekts gemessen.

Für den aktuellen Stand Worker-Zahlen wie `1`, `2`, `4` und `8` mit identischen
Seeds, Populationen und Episodenzahlen vergleichen. `metrics.jsonl` enthält
den Durchsatz; die erste Generation enthält auch den Prozessstart. Kleine
Populationen und kurze Episoden können durch Prozesskommunikation langsamer
werden. Größere Populationen verändern zudem die Evolution und sind kein
reiner Geschwindigkeitsschalter.

Ein erster Benchmark ist in [docs/performance.md](docs/performance.md)
dokumentiert. Er ist keine Messung auf der RTX 5090 oder dem Zielrechner.

## Tests

```bash
python -B -m unittest discover -s tests -v
```

Die Logiktests benötigen keine zusätzlichen Pakete. Mit installiertem
`neat-python` kommen Integrationstests für serielle/parallele Gleichheit und
Checkpoint-Wiederaufnahme hinzu. Mit Pygame wird auch die Gleichheit zwischen
Anzeige und Headless-Modus mit SDL-Dummy-Ausgabe geprüft. Die Logiktests
verwenden die echte Simulation ohne Grafik-Mocks.

## Dateien

| Datei | Aufgabe |
| --- | --- |
| `Snake_Game.py` | Einstiegspunkt und CLI-Aufruf |
| `training.py` | Evolution, Optionen, Metriken und Checkpoints |
| `evaluation.py` | Episoden, Fitness und dauerhafter CPU-Prozesspool |
| `rendering.py` | Optionale Pygame-Anzeige |
| `c_Game.py` | Gemeinsame Simulation und Beobachtungen |
| `c_Snake.py`, `c_Bait.py`, `c_Playboard.py` | Schlange, Futter und Darstellung |
| `config-feedforward.txt` | NEAT-Parameter |
| `Requirements-headless.txt` | Abhängigkeit für Training ohne Anzeige |
| `Requirements.txt` | Abhängigkeiten einschließlich Anzeige |
