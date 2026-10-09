import numpy as np
from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from config import units
from atmosphere import isa


def test_freefall_analytic_small_dt():
    h0 = 100.0
    s = State(pos=np.array([0.0, 0.0, h0]), vel=np.zeros(3))
    m = MassModel(dry_mass_kg=500.0, fuel_kg=0.0)
    tf = 1.0
    dt = 0.01
    sim = Simulator(SimConfig(t0=0.0, tf=tf, dt=dt, seed=42))
    hist = sim.run(s.copy(), m)
    h_num = hist[-1].height_m
    h_an = h0 - 0.5 * units.GRAVITY_M_S2 * tf**2
    v_an = -units.GRAVITY_M_S2 * tf
    assert np.isclose(hist[-1].vel[2], v_an, atol=1e-2)
    n = int(tf / dt)
    h_an_euler = h0 + (-units.GRAVITY_M_S2) * dt * dt * n * (n + 1) / 2.0
    assert np.isclose(h_num, h_an_euler, atol=1e-3)
    assert hist[-1].mass_kg == 500.0


def test_no_force_constant_velocity():
    s = State(pos=np.array([1.0, 2.0, 3.0]), vel=np.array([2.0, -1.0, 0.5]))
    m = MassModel(dry_mass_kg=100.0, fuel_kg=0.0)
    sim = Simulator(SimConfig(tf=2.0, dt=0.005, seed=1, gravity=False))
    hist = sim.run(s.copy(), m)
    assert np.allclose(hist[-1].vel, s.vel, atol=1e-12)
    assert np.allclose(hist[-1].pos, s.pos + s.vel * 2.0, atol=1e-10)


def test_mass_burn_and_sync():
    m = MassModel(dry_mass_kg=400.0, fuel_kg=100.0)
    burned = m.burn(10.0, 2.0)
    assert burned == 20.0
    assert m.fuel_kg == 80.0
    assert m.total_mass() == 480.0
    try:
        MassModel(-1, 0)
        assert False
    except ValueError:
        pass


def test_isa_monotonic_with_height():
    rho0 = isa.density(0.0)
    rho1k = isa.density(1000.0)
    rho5k = isa.density(5000.0)
    assert rho0 > rho1k > rho5k


def test_sensor_reproducible_with_seed():
    from sensors.base import Sensor
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    s1 = Sensor("acc", std=0.5, bias=0.1)
    s2 = Sensor("acc", std=0.5, bias=0.1)
    v1 = s1.measure(0.0, 0.0, rng=rng1)
    v2 = s2.measure(0.0, 0.0, rng=rng2)
    assert v1 == v2
