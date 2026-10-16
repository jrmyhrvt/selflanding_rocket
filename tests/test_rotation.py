import numpy as np
from state.core import State
from physics import rotation
from simulation.loop import Simulator, SimConfig
from mass_model.mass import MassModel


def test_quaternion_helpers():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rotation.norm(q), 1.0)
    qn = rotation.normalize(q)
    assert np.allclose(qn, q)


def test_integrate_rotation_no_change_when_omega_zero():
    s = State(pos=[0, 0, 10], q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.zeros(3), alpha=np.zeros(3))
    q0 = s.q.copy()
    rotation.integrate_rotation(s, 0.01)
    assert np.allclose(s.q, q0)
    assert np.allclose(rotation.norm(s.q), 1.0)


def test_integrate_rotation_simple_rotation():
    omega = np.array([0.0, 0.0, 1.0])
    s = State(q=np.array([1.0, 0.0, 0.0, 0.0]), omega=omega.copy(), alpha=np.zeros(3))
    rotation.integrate_rotation(s, 0.5)
    assert np.allclose(rotation.norm(s.q), 1.0, atol=1e-10)


def test_rotation_disabled_by_default():
    s = State(pos=[0, 0, 5], q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.array([0.0, 0.0, 2.0]), alpha=np.zeros(3))
    m = MassModel(dry_mass_kg=100, fuel_kg=0)
    sim = Simulator(SimConfig(tf=0.2, dt=0.1, gravity=False))
    h = sim.run(s.copy(), m)
    assert np.allclose(h[0].q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(h[-1].q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rotation.norm(h[-1].q), 1.0)


def test_rotation_enabled_changes_q():
    s = State(pos=[0, 0, 5], q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.array([0.0, 0.0, 1.0]), alpha=np.zeros(3))
    m = MassModel(dry_mass_kg=100, fuel_kg=0)
    cfg = SimConfig(tf=0.2, dt=0.1, gravity=False, integrate_rotation=True)
    sim = Simulator(cfg)
    h = sim.run(s.copy(), m)
    assert not np.allclose(h[-1].q, h[0].q)
    assert np.allclose(rotation.norm(h[-1].q), 1.0, atol=1e-10)
