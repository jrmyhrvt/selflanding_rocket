import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from physics.controller import LandingController
from visualization.viz3d_ctrl import CtrlViz3D
from evaluation import landing as L


def _simulate(target_xyz):
    s = State(pos=np.array([0.0, 0.0, 15.0]))
    m = MassModel(300.0, 0.0)
    cfg = SimConfig(tf=120.0, dt=0.05, stop_at_ground=True, ground_z=0.0)
    sim = Simulator(cfg, guidance=LandingController(descent_speed_m_s=0.5))
    return sim.run(s.copy(), m, target=np.asarray(target_xyz, dtype=float))


def _make(target_xy):
    target = np.array([target_xy[0], target_xy[1], 0.0])
    hist = _simulate(target)
    viz = CtrlViz3D(hist, target_pos=target, guidance_active=True, resimulate=_simulate)
    return viz, hist


def test_viz_builds_with_guidance_flag():
    viz, _ = _make((2.0, 1.0))
    assert viz.guidance_active is True
    plt.close(viz.fig)


def test_slider_resimulates_and_moves_trajectory():
    viz, hist_before = _make((2.0, 1.0))
    end_before = hist_before[-1].pos.copy()
    viz.sx.set_val(-4.0)
    viz.sy.set_val(1.5)
    end_after = viz.history[-1].pos.copy()
    # A genuinely new simulation was produced (not the old replay).
    assert viz.history is not hist_before
    assert not np.allclose(end_before, end_after)
    # And it lands at the new target.
    res = L.evaluate_landing(viz.history, viz.target)
    assert res["status"] == L.LANDED
    assert res["dist_xy"] <= L.DIST_TOL_M
    assert abs(viz.target[0] - (-4.0)) < 1e-9
    assert abs(viz.target[1] - 1.5) < 1e-9
    plt.close(viz.fig)


def test_without_resimulate_is_replay_only():
    target = np.array([2.0, 1.0, 0.0])
    hist = _simulate(target)
    viz = CtrlViz3D(hist, target_pos=target, guidance_active=True)  # no resimulate
    viz.sx.set_val(-4.0)
    # Replay-only: the stored history is unchanged.
    assert viz.history is hist
    plt.close(viz.fig)
