from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from evaluation import export, reports
from visualization import timeplots
import os


def main():
    outdir = "output"
    os.makedirs(outdir, exist_ok=True)
    s = State(pos=[0, 0, 100])
    m = MassModel(dry_mass_kg=500, fuel_kg=100)
    sim = Simulator(SimConfig(tf=2.0, dt=0.01))
    hist = sim.run(s.copy(), m)

    csv_path = os.path.join(outdir, "trajectory.csv")
    json_path = os.path.join(outdir, "summary.json")
    h_path = os.path.join(outdir, "height.png")
    v_path = os.path.join(outdir, "velocity.png")

    export.history_to_csv(hist, csv_path)
    reports.summarize(hist, json_path)
    timeplots.plot_height(hist, h_path)
    timeplots.plot_velocity(hist, v_path)

    print(f"steps={len(hist)} final_h={hist[-1].height_m:.2f}")
    print("Output written to:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print(f"  {h_path}")
    print(f"  {v_path}")


if __name__ == "__main__":
    main()
