"""Isolated virtual rigid-body rotation demo/tests.

Uses the existing Simulator + RigidBody + integrate_rotation_body with a
constant, prescribed body-frame moment. Rotation is enabled ONLY here via
the opt-in SimConfig flag; no scenario/controller defaults are changed.

NOT for real hardware. Learning simulation only.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from state.core import State
from mass_model.mass import MassModel
from physics.rigidbody import RigidBody
from physics.rotation import norm as qnorm
from simulation.loop import Simulator, SimConfig


def run_rotation_demo(moment_body, omega0_body=None, tf=1.0, dt=0.01,
                      I_diag=(1.0, 2.0, 3.0), start_height_m=10.0):
    """Run a pure rotation history (translation kept trivial, gravity off)."""
    rb = RigidBody(mass_kg=100.0, I_diag=list(I_diag))
    state = State(
        pos=np.array([0.0, 0.0, start_height_m]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_b=np.zeros(3) if omega0_body is None else np.asarray(omega0_body, float),
        moment_total=np.asarray(moment_body, float),
    )
    cfg = SimConfig(tf=tf, dt=dt, gravity=False,
                    integrate_rotation_body=True, rigid_body=rb)
    sim = Simulator(cfg)
    mass = MassModel(dry_mass_kg=100.0, fuel_kg=0.0)
    return sim.run(state, mass)


def test_zero_moment_keeps_upright():
    history = run_rotation_demo(moment_body=[0.0, 0.0, 0.0])
    for h in history:
        assert np.allclose(h.q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(qnorm(history[-1].q), 1.0)


def test_constant_moment_causes_rotation():
    history = run_rotation_demo(moment_body=[3.0, 0.0, 0.0])
    assert not np.allclose(history[-1].q, [1.0, 0.0, 0.0, 0.0])
    # q about x only -> x-component becomes non-zero, y/z stay zero
    assert abs(history[-1].q[1]) > 1e-3
    assert np.allclose(history[-1].q[2:], [0.0, 0.0])


def test_rotation_quaternions_stay_normalized():
    hist = run_rotation_demo(moment_body=[0.5, 0.0, 0.0], tf=2.0, dt=0.005)
    assert all(abs(qnorm(s.q) - 1.0) < 1e-9 for s in hist)


def test_rotation_grows_with_moment():
    small = run_rotation_demo(moment_body=[1.0, 0.0, 0.0])
    big = run_rotation_demo(moment_body=[5.0, 0.0, 0.0])
    # angle from identity (q_x component) should be larger for bigger moment
    assert abs(big[-1].q[1]) > abs(small[-1].q[1])


def test_history_visualizable_in_studio():
    """Existing 3D studio can show this history via its public entry."""
    import matplotlib.pyplot as plt
    from visualization.viz3d_studio import LandingStudio

    hist = run_rotation_demo(moment_body=[0.0, 4.0, 0.0], tf=1.0, dt=0.02)
    studio = LandingStudio(history=hist)
    try:
        assert len(studio.history) == len(hist)
        studio.update(len(hist) - 1)
        # tilted: body axis no longer purely vertical
        assert not np.allclose(studio.body_axis_world(), [0.0, 0.0, 1.0])
    finally:
        plt.close(studio.fig)
