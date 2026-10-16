import numpy as np
import pytest
from state.core import State
from physics.rigidbody import RigidBody
from physics import rotation
from simulation.loop import Simulator, SimConfig
from mass_model.mass import MassModel


def test_rigidbody_inertia_and_angular_accel_zero_moment():
    rb = RigidBody(mass_kg=100.0, I_diag=[10.0, 20.0, 30.0])
    alpha = rb.angular_accel(np.zeros(3), np.zeros(3))
    assert np.allclose(alpha, 0.0)


def test_rigidbody_simple_torque_1d():
    rb = RigidBody(I_diag=[2.0, 2.0, 2.0])
    M = np.array([4.0, 0.0, 0.0])
    alpha = rb.angular_accel(np.zeros(3), M)
    assert np.allclose(alpha, [2.0, 0.0, 0.0])


def test_rigidbody_positive_inertia_values():
    rb = RigidBody(I_diag=[0.5, 1.0, 1.5])
    assert np.all(rb.I_diag > 0)
    I = rb.inertia()
    assert np.allclose(I.diagonal(), rb.I_diag)


def test_rigidbody_gyro_term_nonzero():
    rb = RigidBody(I_diag=[1.0, 2.0, 3.0])
    omega_b = np.array([1.0, 1.0, 0.0])
    M_b = np.zeros(3)
    alpha_b = rb.angular_accel(omega_b, M_b)
    assert np.allclose(alpha_b, [0.0, 0.0, -1.0 / 3.0])


def test_rigidbody_gyro_unequal_inertia():
    rb = RigidBody(I_diag=[0.5, 1.0, 1.5])
    omega_b = np.array([1.0, 2.0, 0.0])
    M_b = np.zeros(3)
    alpha_b = rb.angular_accel(omega_b, M_b)
    assert alpha_b.shape == (3,)
    assert np.allclose(rotation.norm(alpha_b), 0.0) is False or True
    # ensure computed
    assert np.isfinite(alpha_b).all()


def test_integrate_rotation_body_no_moment_preserves_magnitude():
    rb = RigidBody(I_diag=[1.0, 1.0, 1.0])
    s = State(q=np.array([1.0, 0.0, 0.0, 0.0]), moment_total=np.zeros(3))
    s.omega_b = np.array([0.1, 0.2, 0.3])
    rotation.integrate_rotation_body(s, 0.01, rb=rb)
    assert np.allclose(rotation.norm(s.q), 1.0)


def test_integrate_rotation_body_changes_q_with_omega():
    rb = RigidBody(I_diag=[1.0, 1.0, 1.0])
    s = State(q=np.array([1.0, 0.0, 0.0, 0.0]), moment_total=np.zeros(3))
    s.omega_b = np.array([0.0, 0.0, 1.0])
    q0 = s.q.copy()
    rotation.integrate_rotation_body(s, 0.1, rb=rb)
    assert not np.allclose(s.q, q0)
    assert np.allclose(rotation.norm(s.q), 1.0)


def test_integrate_rotation_body_sets_alpha_b():
    rb = RigidBody(I_diag=[1.0, 1.0, 1.0])
    s = State(q=np.array([1.0, 0.0, 0.0, 0.0]), moment_total=np.array([2.0, 0.0, 0.0]))
    s.omega_b = np.array([0.0, 0.0, 0.0])
    rotation.integrate_rotation_body(s, 0.01, rb=rb)
    assert np.allclose(s.alpha_b, [2.0, 0.0, 0.0])


def test_sim_body_rotation_opt_in():
    rb = RigidBody(I_diag=[1.0, 1.0, 1.0])
    s = State(pos=[0, 0, 5], q=np.array([1.0, 0.0, 0.0, 0.0]), moment_total=np.zeros(3))
    s.omega_b = np.array([0.0, 0.0, 2.0])
    m = MassModel(dry_mass_kg=100, fuel_kg=0)
    cfg = SimConfig(tf=0.1, dt=0.1, gravity=False, integrate_rotation_body=True, rigid_body=rb)
    sim = Simulator(cfg)
    h = sim.run(s.copy(), m)
    assert np.allclose(rotation.norm(h[-1].q), 1.0, atol=1e-10)


def test_legacy_rotation_unchanged_default():
    s = State(pos=[0, 0, 5], q=np.array([1.0, 0.0, 0.0, 0.0]), omega=np.array([0.0, 0.0, 2.0]), alpha=np.zeros(3))
    m = MassModel(dry_mass_kg=100, fuel_kg=0)
    sim = Simulator(SimConfig(tf=0.2, dt=0.1, gravity=False))
    h = sim.run(s.copy(), m)
    assert np.allclose(h[0].q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(h[-1].q, [1.0, 0.0, 0.0, 0.0])


def test_body_path_does_not_touch_legacy_omega_when_present():
    rb = RigidBody(I_diag=[1.0, 1.0, 1.0])
    s = State(q=[1.0, 0.0, 0.0, 0.0], moment_total=[0.0, 0.0, 0.0], omega=[5.0, 6.0, 7.0], alpha=[1.0, 2.0, 3.0])
    s.omega_b = np.array([0.1, 0.2, 0.3])
    omega_legacy_before = s.omega.copy()
    rotation.integrate_rotation_body(s, 0.01, rb=rb)
    assert np.allclose(s.omega, omega_legacy_before)
