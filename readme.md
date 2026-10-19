# Snake AI

Snake mit Pygame und neuronalen Netzen, die mit NEAT trainiert werden.
Beim Start steuert die KI automatisch 24 Snake-Spiele parallel. Eine manuelle
Steuerung per Tastatur ist derzeit nicht implementiert.

## Voraussetzungen

- Python 3 mit `pip` und `venv`.
- Eine grafische Desktop-Umgebung, damit Pygame ein Fenster öffnen kann.
- Die Abhängigkeiten aus `Requirements.txt`: `pygame-ce==2.5.7` und `neat-python==0.92`.

Pygame CE wird im Code weiterhin als `pygame` importiert. Ein kurzer echter
NEAT-Trainingslauf wurde mit Python 3.14.8 und diesen Paketversionen unter Linux
mit SDL-Dummy-Videoausgabe geprüft. Das prüft keinen sichtbaren Fensterstart.

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

## Aktuelle Trainingsparameter

| Parameter | Einstellung und Wirkung |
| --- | --- |
| Eingänge / Ausgänge | `5` Eingänge, `1` Ausgang für Links/Geradeaus/Rechts |
| Aktivierung | `tanh`; Ausgang über `0.5`: rechts, unter `-0.5`: links, sonst geradeaus |
| Population | `24`, passend zu den derzeit fest angelegten 24 Spielfeldern |
| Energie | Anfangs `20` Züge, pro Futter zusätzlich `20` |
| Fitness | `+100` pro Futter, `-10` beim Spielende |
| Fitnessziel | `10000`; wegen des Abzugs beim Spielende sind mindestens 101 Futterstücke nötig |
| Generationslimit | `1000`; das Fitnessziel kann das Training deutlich früher beenden |
| Stagnation | `max_stagnation = 1000`; innerhalb des Generationslimits werden stagnierende Arten praktisch nicht aussortiert |

Beim früheren Fitnessziel `1000` wurde das Ziel in zehn kurzen Zufallsstarts
jeweils nach 1–5 Generationen erreicht. Das aktuelle Ziel ist höher; diese
kurzen Tests belegen noch keine zuverlässig gute Strategie auf neuen
Futterfolgen.

## Eingaben der KI

Das Netz erhält fünf Eingaben in dieser Reihenfolge:

| Eingabe | Werte und Bedeutung |
| --- | --- |
| Gefahr vorne | `1`: Der nächste Schritt trifft eine Wand oder den Körper; sonst `0` |
| Gefahr links | Gleiche Prüfung nach einer Linksdrehung |
| Gefahr rechts | Gleiche Prüfung nach einer Rechtsdrehung |
| Futter vor/hinter der Schlange | `1`: vorne, `-1`: hinten, `0`: seitlich auf gleicher Höhe |
| Futter rechts/links der Schlange | `1`: rechts, `-1`: links, `0`: auf der Blickachse |

Alle Richtungen beziehen sich auf die aktuelle Blickrichtung der Schlange.
Die Futtereingaben beschreiben die Richtung, nicht die Entfernung. Das Netz
entscheidet vor jedem Schritt, auch vor dem ersten, ob die Schlange geradeaus
läuft, links oder rechts abbiegt. Die NEAT-Konfiguration verwendet dafür
`num_inputs = 5`.

## Hinweise bei Startproblemen

- **`ModuleNotFoundError` für `pygame` oder `neat`:** Die virtuelle Umgebung
  aktivieren und `python -m pip install -r Requirements.txt` darin ausführen.
- **Installation von Pygame schlägt fehl:** Die aktuelle `Requirements.txt`
  verwenden. Sie enthält Pygame CE anstelle der früher festgelegten
  Pygame-Version `2.0.1`.
- **Kein Fenster / Fehler zur Videoausgabe:** Das Programm in einer grafischen
  Desktop-Sitzung starten.

## Regressionstests

```bash
python3 -B -m unittest discover -s tests -v
```

Die Tests benötigen keine zusätzlichen Pakete. Pygame und NEAT werden für die
Logiktests durch Test-Doubles ersetzt; sie prüfen keine Grafik oder tatsächliche
NEAT-Evolution. Geprüft werden Futterplatzierung, Richtungswechsel,
Wandabstände, die fünf KI-Eingaben und das Entfernen beendeter Spiele inklusive
Fitness-Zuordnung.

Die weiterhin separat getestete Wandabstandsfunktion beschreibt die Strecke
bis zur letzten sicheren Zelle in Blickrichtung, links und rechts. Sie wird
nicht mehr als KI-Eingabe verwendet.

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
