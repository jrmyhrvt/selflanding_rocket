"""
Diagnose der estimation-Modus-Fehlschlaege (TOO_FAST) im Sensor-Experiment.

Zeigt fuer jede fehlgeschlagene virtuelle Landung den zeitlichen Verlauf von
wahrer Geschwindigkeit, verrauschter Messung und geschaetzter Geschwindigkeit
und quantifiziert, WANN die Schaetzung von der Wahrheit abweicht (Schaetz-Lag).

Es werden dieselben Szenarien und Seeds wie im Sensor-Experiment verwendet und
die laufende Simulation zeichnet die Zeitreihen live auf
(`ObservedController.record`). Keine kuenstlichen Beispielkurven.

Keine Parameteroptimierung, keine Controller-Aenderung. Rein virtuell.

Aufruf:
    PYTHONPATH=. python3 landing_sensor_diagnostics_app.py
    PYTHONPATH=. python3 landing_sensor_diagnostics_app.py --show
    PYTHONPATH=. python3 landing_sensor_diagnostics_app.py --outdir output/diag
    PYTHONPATH=. python3 landing_sensor_diagnostics_app.py --seeds 1234,202
"""
import argparse

from evaluation.sensor_diagnostics import main, SE


def build_parser():
    p = argparse.ArgumentParser(description="Schaetz-Lag Diagnose")
    p.add_argument("--outdir", default="output/sensor_diagnostics",
                   help="Ausgabeordner fuer CSV und Plots")
    p.add_argument("--show", action="store_true",
                   help="Plotfenster interaktiv anzeigen statt nur speichern")
    p.add_argument("--seeds", default=",".join(str(s) for s in SE.DEFAULT_SEEDS),
                   help="Komma-getrennte Seeds, z. B. 1234,202")
    return p


def run(args=None):
    seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
    return main(outdir=args.outdir, show=args.show, seeds=seeds)


if __name__ == "__main__":
    run(build_parser().parse_args())