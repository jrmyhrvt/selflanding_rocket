"""
Entry point for the RANDOM-SCENARIO mode of the unified 3D flight simulation.

Default: run a whole SERIES (default 100) of randomly sampled, physically valid
initial conditions through the existing 3D physics + automatic controller, assess
each landing with the existing criteria, and write CSV / JSON / plot under
`output/random_scenario/`. No 100 windows are opened.

With `--visualize` a SINGLE sampled scenario is shown live in the existing 3D
view (visualization/flight_view.py) so one random run can be inspected.

The interactive single-run app `flight_sim_app.py` is untouched.

Learning simulation only. NOT for real hardware.

Usage:
    PYTHONPATH=. python3 random_scenario_app.py                 # 100 runs, headless
    PYTHONPATH=. python3 random_scenario_app.py --runs 50 --seed 1234
    PYTHONPATH=. python3 random_scenario_app.py --no-sensor
    PYTHONPATH=. python3 random_scenario_app.py --visualize --seed 7
"""
import argparse

import numpy as np

from evaluation import random_scenario as RS
from evaluation.random_scenario import RandomScenarioConfig


def _visualize_one(cfg: RandomScenarioConfig, seed: int) -> None:
    """Show ONE sampled random scenario live in the existing 3D flight view."""
    from visualization.flight_view import FlightView

    rng = np.random.default_rng(seed)
    ic = RS.sample_initial_state(rng, cfg)
    flight_cfg = RS.build_flight_config(cfg, ic)
    target = RS.target_point(cfg, ic)
    print(f"Visualisiere Einzelszenario (Seed {seed}):")
    print(f"  Start  pos={np.round(ic['pos'], 2)}  vel={np.round(ic['vel'], 2)} "
          f"m/s  tilt0={RS.initial_tilt_deg(ic):.1f} deg  "
          f"omega_b={np.round(ic['omega_b'], 3)} rad/s")
    src = "Startposition" if cfg.target_from_start else "target_center_xy (fest)"
    print(f"  Ziel   = ({target[0]:.2f}, {target[1]:.2f}) m  [{src}]")
    print(f"  Positionsregelung kp_xy={cfg.kp_xy}  vh_max={cfg.vh_max_m_s} m/s")
    print("  Steuerung = vorhandener Auto-Regler (model_control), ohne Sensorrauschen")
    FlightView(cfg=flight_cfg).show()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Random-Szenario-Modus (3D-Flug)")
    parser.add_argument("--runs", type=int, default=100,
                        help="Anzahl zufaelliger Laeufe (Default 100)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Basis-Seed fuer exakte Reproduktion (Default: zufaellig)")
    parser.add_argument("--outdir", default="output/random_scenario",
                        help="Zielordner fuer CSV/JSON/Plot")
    parser.add_argument("--no-sensor", action="store_true",
                        help="ohne Sensorrauschen (Regler sieht den wahren Zustand)")
    parser.add_argument("--estimator", action="store_true",
                        help="Low-Pass-Schaetzer im Sensor-/Beobachtungspfad aktivieren")
    parser.add_argument("--visualize", action="store_true",
                        help="EIN zufaelliges Szenario live in 3D zeigen statt der Serie")
    args = parser.parse_args(argv)

    cfg = RandomScenarioConfig(
        n_runs=args.runs, seed=args.seed,
        sensor_enabled=not args.no_sensor,
        estimator_enabled=args.estimator,
    )
    if args.visualize:
        seed = args.seed if args.seed is not None else RS.resolved_seed(cfg)
        _visualize_one(cfg, seed)
        return 0

    RS.main(outdir=args.outdir, cfg=cfg, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
