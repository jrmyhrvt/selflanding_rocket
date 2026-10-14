#!/usr/bin/env python3
"""
Experiment / evaluation mode for the virtual autoland learning sim.

Runs bounded parameter sweeps (start height, target distance, initial
velocities, time step, timeout), prints a result table, writes a CSV and saves
summary plots. Reuses simulation.scenario.run_scenario and
evaluation.landing.evaluate_landing - no second physics implementation.

Run:
    PYTHONPATH=. python3 landing_experiment_app.py

Optional interactive plot window (needs a GUI backend):
    PYTHONPATH=. python3 landing_experiment_app.py --show

Virtual learning tool only. No hardware, no real-rocket control.
"""
import sys
from evaluation.experiment import main


if __name__ == "__main__":
    show = "--show" in sys.argv
    outdir = "output/experiment"
    main(outdir=outdir, show=show)
