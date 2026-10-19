# Snake AI

Snake mit Pygame und neuronalen Netzen, die mit NEAT trainiert werden.
Beim Start steuert die KI automatisch 24 Snake-Spiele parallel. Eine manuelle
Steuerung per Tastatur ist derzeit nicht implementiert.

## Voraussetzungen

- Python 3 mit `pip` und `venv`.
- Eine grafische Desktop-Umgebung, damit Pygame ein Fenster öffnen kann.
- Die Abhängigkeiten aus `Requirements.txt`: `pygame==2.0.1` und `neat-python==0.92`.

**Kompatibilität:** Das Projekt verwendet ältere Paketversionen. Installation
und Ausführung können mit aktuellen Python-Versionen scheitern; siehe die
Hinweise unten.

## Installation und Start

Im Projektordner unter Linux/macOS ausführen:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r Requirements.txt
python Snake_Game.py
```

Unter Windows (PowerShell):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r Requirements.txt
python Snake_Game.py
```

Bei späteren Starts genügt es, die virtuelle Umgebung zu aktivieren und
`python Snake_Game.py` auszuführen. Die Groß-/Kleinschreibung von
`Requirements.txt` und `Snake_Game.py` beachten.

## Was passiert beim Start?

- Ein Fenster mit 1850 × 1000 Pixeln zeigt 24 Spielfelder.
- Die Schlangen bewegen sich automatisch. Futter ist grün, der Kopf rot und
  der Körper weiß.
- Punkte und die aktuelle Generation werden im Fenster angezeigt.
  Trainingsstatistiken erscheinen im Terminal.
- Das Training läuft für höchstens 1000 Generationen und kann beim Erreichen
  des Fitness-Zielwerts aus `config-feedforward.txt` früher enden.
- Zum Beenden das Fenster schließen oder im Terminal `Strg+C` drücken.

Jeder Start beginnt ein neues Training. Das beste Netz wird im aktuellen Code
nicht als Datei gespeichert.

## Hinweise bei Startproblemen

- **`ModuleNotFoundError` für `pygame` oder `neat`:** Die virtuelle Umgebung
  aktivieren und `python -m pip install -r Requirements.txt` darin ausführen.
- **Installation von Pygame schlägt fehl:** Die festgelegte Version `2.0.1`
  ist alt. Für eine aktuelle Python-Umgebung müssen gegebenenfalls die
  Abhängigkeiten und der Code angepasst werden.
- **Kein Fenster / Fehler zur Videoausgabe:** Das Programm in einer grafischen
  Desktop-Sitzung starten.

## Regressionstests

```bash
python3 -B -m unittest discover -s tests -v
```

Die Tests benötigen keine zusätzlichen Pakete. Pygame und NEAT werden für die
Logiktests durch Test-Doubles ersetzt; sie prüfen keine Grafik oder tatsächliche
NEAT-Evolution. Geprüft werden Futterplatzierung, Richtungswechsel,
Wandabstände und das Entfernen beendeter Spiele inklusive Fitness-Zuordnung.

Die Wandabstände beschreiben die Strecke bis zur letzten sicheren Zelle in
Blickrichtung, links und rechts. Ein Wert von `0` bedeutet, dass der nächste
Schritt in dieser Richtung die Wand treffen würde.

## Projektdateien

| Datei | Aufgabe |
| --- | --- |
| `Snake_Game.py` | Einstiegspunkt, NEAT-Training und Verwaltung der 24 Spiele |
| `c_Game.py` | Spiellogik, Punkte und Eingaben für das neuronale Netz |
| `c_Snake.py` | Bewegung, Wachstum, Energie und Kollisionen der Schlange |
| `c_Bait.py` | Platzierung und Darstellung des Futters |
| `c_Playboard.py` | Darstellung des Spielfelds |
| `config-feedforward.txt` | NEAT-Konfiguration, unter anderem Population und Fitness-Ziel |
| `Requirements.txt` | Festgelegte Python-Abhängigkeiten |
