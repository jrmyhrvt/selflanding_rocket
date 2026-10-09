from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
import numpy as np

def test_mass_consistency():
    m = MassModel(500, 100)
    assert m.total_mass() == 600

def test_freefall_basic():
    s = State(pos=np.array([0,0,100.0]), vel=np.zeros(3))
    m = MassModel(500, 100)
    sim = Simulator(SimConfig(tf=1.0, dt=0.01, seed=1))
    hist = sim.run(s.copy(), m)
    assert hist[-1].height_m < hist[0].height_m
    assert len(hist) == 100
