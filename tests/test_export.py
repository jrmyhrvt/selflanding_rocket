import os
import tempfile
import numpy as np
from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from evaluation import export, reports


def test_history_to_csv_basic():
    s = State(pos=np.array([0.0, 1.0, 100.0]), vel=np.array([0.0, 0.0, -5.0]))
    m = MassModel(500.0, 50.0)
    hist = Simulator(SimConfig(tf=0.2, dt=0.1, seed=1)).run(s.copy(), m)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "traj.csv")
        res = export.history_to_csv(hist, p)
        assert os.path.exists(res)
        with open(res) as f:
            lines = f.readlines()
        assert lines
        assert lines[0].startswith("t,pos_x,pos_y,pos_z")


def test_summarize_nonempty():
    s = State(pos=np.array([0.0, 0.0, 50.0]))
    m = MassModel(300.0, 0.0)
    hist = Simulator(SimConfig(tf=0.1, dt=0.05, seed=2)).run(s.copy(), m)
    data = reports.summarize(hist)
    assert data["count"] == 2
    assert data["height0"] > data["heightf"]


def test_summarize_empty():
    data = reports.summarize([])
    assert data["count"] == 0
