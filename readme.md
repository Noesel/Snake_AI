# Snake AI mit TensorNEAT

Snake trainiert mit [TensorNEAT](https://github.com/EMI-Group/tensorneat) und
JAX. Netze, Evolution und Snake-Episoden werden als Array-Operationen kompiliert
und auf einem ausgewählten CPU- oder GPU-Gerät ausgeführt. Die fünf
Must-have-Eingaben bleiben erhalten.

## Installation

Getestet unter Linux mit **Python 3.12.15**, **JAX 0.11.2**, TensorNEAT-Revision
`ee204900e8e55ffe0ed756651ea871c70230dd34` und optional Pygame CE 2.5.7.
`constraints.txt` hält die getesteten Abhängigkeiten fest. TensorNEAT bringt
zusätzliche Pakete wie Brax und MuJoCo über seine eigenen Abhängigkeiten mit.

Eine neue Umgebung verwenden:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Für CPU-Training ohne Anzeige:

```bash
python -m pip install -r Requirements-headless.txt
```

Für CPU-Training mit Vorschau stattdessen:

```bash
python -m pip install -r Requirements.txt
```

### RTX 5090 / NVIDIA-GPU unter Linux

```bash
python -m pip install -r Requirements-gpu.txt
python -c "import jax; print(jax.devices())"
```

Das GPU-Paket verwendet JAX mit CUDA 13. Laut
[JAX-Installationsanleitung](https://docs.jax.dev/en/latest/installation.html)
benötigt CUDA 13 unter Linux einen NVIDIA-Treiber ab Version 580. Die
CUDA-Laufzeitbibliotheken werden durch das JAX-Extra installiert. Linux ist der
vorgesehene GPU-Weg; JAX führt WSL2-GPU-Unterstützung als experimentell und
unterstützt NVIDIA-GPUs nicht unter nativem Windows.

`--device gpu` meldet einen Fehler, wenn kein GPU-Backend verfügbar ist.
`--device auto` verwendet das bevorzugte verfügbare Gerät; das kann die CPU sein.
Die GPU-Installation und Leistung auf der RTX 5090 sind hier noch nicht getestet.

## Start

Kleiner CPU-Funktionstest:

```bash
python Snake_Game.py --headless --device cpu --population 8 --episodes 2 --max-steps 40 --max-nodes 8 --max-conns 8 --generations 3
```

Beispiel für einen größeren GPU-Lauf:

```bash
python Snake_Game.py --headless --device gpu --population 1024 --episodes 5 --generations 1000 --seed 42
```

Der erste Start kompiliert das Programm. Änderungen an Population, Episodenzahl,
Kapazitäten oder Batchgröße können eine neue Kompilierung auslösen. Ein
persistenter JAX-Kompilierungscache liegt unter `runs/.jax-cache/`.

Ohne `--headless` wird nach jeder Generation eine Episode des bisher besten
Netzes angezeigt:

```bash
python Snake_Game.py --device cpu --population 32 --generations 100
```

Für eine Vorschau nach GPU-Installation zusätzlich `pygame-ce==2.5.7`
installieren. `--fps 0` hebt das Vorschau-FPS-Limit auf. Fenster schließen oder
`Strg+C` beendet den Lauf und speichert den letzten abgeschlossenen Zustand.
Die Vorschau ist eine einzelne Schlange; die Population wird gebündelt ohne
Zeichnen trainiert. Die Vorschau kostet zusätzliche Zeit.

## Parallelität und Speicher

JAX bündelt die Population und die Episoden mit `vmap`; die Spielschleifen,
Netzauswertung und Evolution werden mit XLA kompiliert. Während des Headless-
Trainings gibt es keine Python-Schleife oder CPU-GPU-Übertragung pro Spielzug.
Der Host liest nach jeder Generation Metriken und gegebenenfalls das beste Netz.

Der Körper liegt in einem festen Ringpuffer, ergänzt durch eine Belegungskarte.
Das Spielfeld hat 21 × 21 begehbare Zellen. Abgeschlossene Episoden bleiben
innerhalb eines Batches eingefroren, während andere Episoden weiterlaufen.

`--workers` entfällt. Statt CPU-Prozessen werden Gerät und Batchgröße gewählt:

```bash
python Snake_Game.py --headless --device gpu --population 1024 --batch-size 128 --episodes 5
```

Die Batchgröße muss die Population ohne Rest teilen. Kleinere Batches senken
Speicherspitzen und werden nacheinander ausgewertet; sie sind kein garantierter
Geschwindigkeitsgewinn. Ohne Angabe wird die gesamte Population gebündelt.
Das Backend verwendet derzeit **ein Gerät**, keine verteilte Multi-GPU-Auswertung.

Mit `--max-nodes` und `--max-conns` werden feste Tensor-Kapazitäten definiert.
Größere Werte erlauben komplexere Netze und erhöhen Speicher- und Rechenbedarf.
`max_nodes` zählt auch die fünf Eingabe- und die eine Ausgabezelle mit.
Die automatische GPU-Speichervorbelegung wird standardmäßig deaktiviert;
eine selbst gesetzte Umgebungsvariable `XLA_PYTHON_CLIENT_PREALLOCATE` bleibt
maßgeblich.

## Parameter und Eingaben

Die Standardwerte stehen in `config-tensorneat.json`:

| Parameter | Standard |
| --- | --- |
| Population / Episoden | `256` / `5` |
| Maximaler Episodenumfang | `10000` Schritte |
| Netzkapazität | `16` Knoten, `48` Verbindungen |
| Artenkapazität | Bis zu `10`, bei kleinen Populationen begrenzt |
| Fitnessziel | `10000`, nur mit `--stop-at-target` aktiv |
| Stagnation | `1000`, aus dem bisherigen Ansatz übernommen |
| Laufseed | `42` |

Die angeforderten `--generations` laufen regulär vollständig. Nur
`--stop-at-target` aktiviert den frühen Abbruch. Pro Episode gilt `+100` pro
Futter und `-10` am Ende, auch beim Schrittlimit. Die Fitness wird über Episoden
gemittelt. Startenergie ist `20`, pro Futter kommen `20` hinzu. Ein volles
Spielfeld beendet die Episode ohne Endlossuche nach Futter.

| Eingang | Bedeutung |
| --- | --- |
| 1–3 | Gefahr vorne/links/rechts durch Wand oder Körper: `0` oder `1` |
| 4 | Futter vor/hinter der Schlange: `1`, `-1`, seitlich `0` |
| 5 | Futter rechts/links: `1`, `-1`, auf der Blickachse `0` |

Ein Ausgang mit `tanh`: über `0.5` rechts, unter `-0.5` links, sonst geradeaus.
Die Entscheidung erfolgt vor jedem Zug. Die verlassene Schwanzzelle ist frei;
bei Wachstum wird neues Futter anschließend auf einer freien Zelle platziert.

Alle Kandidaten werden mit denselben festen Episoden-Seeds bewertet. Die
Evolution hat einen getrennten Zufallszustand. Eine spätere Prüfung auf neuen
Seeds bleibt nötig, um Überanpassung zu erkennen. Die Migration ändert
Zufallsfolgen und Evolutionsdetails; frühere Trainingsverläufe werden dadurch
nicht identisch reproduziert.

## Ergebnisse und Fortsetzen

Standardmäßig entsteht für jeden Lauf ein neues Verzeichnis unter `runs/`.
`--output runs/mein-lauf` wählt einen eigenen Ort.

| Datei | Inhalt |
| --- | --- |
| `best.npz` | Beste Knoten-/Verbindungsarrays, Fitness, Generation und Konfiguration |
| `checkpoint.pkl.gz` | Vollständiger TensorNEAT-Zustand einschließlich Zufallsschlüsseln und bestem Netz |
| `metrics.jsonl` | Fitness, Punkte, tatsächlich ausgeführte Schritte, Lauf- und Kompilierungszeiten |
| `run.json` | Konfiguration, Framework-Revision und ausgewähltes Gerät |

Checkpoints werden alle zehn Generationen und am Ende gespeichert.
`--checkpoint-every 1` speichert nach jeder Generation; `0` nur am Ende.
Die Dateien werden über temporäre Dateien ersetzt.

```bash
python Snake_Game.py --headless --device gpu --resume runs/mein-lauf/checkpoint.pkl.gz --generations 1000
```

Damit werden 1000 weitere Generationen gestartet. Konfiguration und Seeds
kommen aus dem Checkpoint; das Gerät und die Anzeige dürfen geändert werden.
Beim Resume müssen JAX-Version und TensorNEAT-Revision übereinstimmen.
Ergebnisse auf unterschiedlichen Gerätetypen müssen nicht bitweise identisch sein.

**Alte neat-python-Checkpoints und `best.pkl` sind mit diesem Backend nicht
kompatibel.** Das frühere Backend bleibt im Git-Verlauf erhalten. Die neue
Konfiguration ist JSON; `config-feedforward.txt` wird nicht mehr verwendet.

## Tests und Messungen

```bash
python -B -m unittest discover -s tests -v
```

Mit installierten Abhängigkeiten werden die echte JAX-Simulation und
TensorNEAT getestet: Spielregelvergleich mit der Python-Referenz, Wachstum,
Schwanzzelle, vollständiges Spielfeld, einzelne/gebündelte Auswertung,
Speicher-Batches, Checkpoints, CLI und Vorschau. Ohne JAX/TensorNEAT werden die
entsprechenden Tests übersprungen; die Referenztests laufen weiterhin.

Lokal wurden 40 Tests mit der CPU bestanden, einschließlich Pygame-Vorschau
mit SDL-Dummy-Ausgabe. Ein Trainingslauf mit 256 Genomen und fünf Episoden über
drei Generationen lief ebenfalls durch. Dieses GPU-orientierte Backend war
auf der lokalen CPU langsamer als der frühere Python-Prozesspool. Ein
RTX-5090-Benchmark steht aus. Details und historische Messungen stehen in
[docs/performance.md](docs/performance.md).

## Dateien

| Datei | Aufgabe |
| --- | --- |
| `Snake_Game.py` | CLI-Einstieg |
| `training.py` | Gerätewahl, Trainingslauf, Metriken und Checkpoints |
| `configuration.py`, `config-tensorneat.json` | Geprüfte Konfiguration |
| `tensor_backend.py` | TensorNEAT und gebündelte Episodenbewertung |
| `jax_snake.py` | Reine JAX-Simulation |
| `rendering.py` | Vorschau des besten Netzes |
| `evaluation.py`, `c_*.py` | Python-Referenz und Zeichenfunktionen |
| `Requirements-headless.txt` | CPU-Installation ohne Anzeige |
| `Requirements-gpu.txt` | NVIDIA-/CUDA-Installation unter Linux |
| `Requirements.txt` | CPU-Installation einschließlich Vorschau |
| `constraints.txt` | Getestete Paketversionen |
