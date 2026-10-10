#!/usr/bin/env python3
"""
Simple entry point: run simulation -> export CSV + JSON summary -> generate time plots.
"""
import os
import argparse

from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from evaluation import export, reports
from visualization import timeplots


def main():
    parser = argparse.ArgumentParser(description="Self-landing rocket sim quick run")
    parser.add_argument("--tf", type=float, default=2.0, help="end time [s]")
    parser.add_argument("--dt", type=float, default=0.01, help="timestep [s]")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--h0", type=float, default=100.0, help="initial height [m]")
    parser.add_argument("--outdir", type=str, default="data", help="output dir")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    s = State(pos=[0.0, 0.0, args.h0])
    m = MassModel(dry_mass_kg=500.0, fuel_kg=100.0)
    sim = Simulator(SimConfig(t0=0.0, tf=args.tf, dt=args.dt, seed=args.seed))
    hist = sim.run(s.copy(), m)

    csv_path = os.path.join(args.outdir, "trajectory.csv")
    json_path = os.path.join(args.outdir, "summary.json")
    hpng = os.path.join(args.outdir, "height.png")
    vpng = os.path.join(args.outdir, "velocity.png")
    ppng = os.path.join(args.outdir, "position.png")

    export.history_to_csv(hist, csv_path)
    reports.summarize(hist, json_path)
    timeplots.plot_height(hist, hpng)
    timeplots.plot_velocity(hist, vpng)
    timeplots.plot_position_components(hist, ppng)

    print("Done.")
    print(f"CSV:       {csv_path}")
    print(f"Summary:   {json_path}")
    print(f"Plots:     {hpng}, {vpng}, {ppng}")


if __name__ == "__main__":
    main()
