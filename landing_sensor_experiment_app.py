"""
Start der Sensor-/Schaetzer-Vergleichsexperimente.

Vergleicht vier Gruppen bei identischen Startbedingungen:
    ideal       Regelung sieht den WAHREN Zustand
    noise       verrauschte Messungen, kein Schaetzer
    dropout     verrauscht + gelegentlich fehlende Messungen
    estimation  verrauscht + fehlend + LowPass-Schaetzer

Alles rein virtuell. Kein Hardware-Zugriff.

Fuer jeden Seed laeuft derselbe Satz Szenarien mit identischen
Anfangsbedingungen; nur der Sensor-Zufallsseed aendert sich. So laesst sich die
Streuung der Ergebnisse abschaetzen.

Aufruf:
    PYTHONPATH=. python3 landing_sensor_experiment_app.py
    PYTHONPATH=. python3 landing_sensor_experiment_app.py --show
    PYTHONPATH=. python3 landing_sensor_experiment_app.py --outdir output/mein_lauf
    PYTHONPATH=. python3 landing_sensor_experiment_app.py --seeds 1234,7,42
"""
import argparse

from evaluation.sensor_experiment import main, DEFAULT_SEEDS


def build_parser():
    p = argparse.ArgumentParser(description="Sensor-/Schaetzer-Vergleich")
    p.add_argument("--outdir", default="output/sensor_experiment",
                   help="Ausgabeordner fuer CSV und Plot")
    p.add_argument("--show", action="store_true",
                   help="Plotfenster interaktiv anzeigen statt nur speichern")
    p.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="Komma-getrennte Seeds, z. B. 1234,7,42")
    return p


def run(args=None):
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
    return main(outdir=args.outdir, show=args.show, seeds=seeds)


if __name__ == "__main__":
    run(build_parser().parse_args())
