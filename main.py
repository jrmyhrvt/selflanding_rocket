from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig

def main():
    s = State(pos=[0,0,100])
    m = MassModel(dry_mass_kg=500, fuel_kg=100)
    sim = Simulator(SimConfig(tf=2.0, dt=0.01))
    hist = sim.run(s.copy(), m)
    print(f"steps={len(hist)} final_h={hist[-1].height_m:.2f}")

if __name__ == "__main__":
    main()
