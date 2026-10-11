import numpy as np
from state.core import State
from system_logic.target import Target
from physics.controller import LandingController


def test_target_distance():
    t = Target(np.array([2.0, 1.0, 0.0]))
    s = State(pos=np.array([0.0, 0.0, 50.0]))
    assert np.isclose(t.distance_xy(s.pos), np.sqrt(5.0))
    assert t.distance_3d(s.pos) > t.distance_xy(s.pos)


def test_controller_returns_accel_shape():
    c = LandingController(descent_speed_m_s=0.5)
    s = State(pos=np.array([0.0, 0.0, 80.0]), vel=np.array([0.0, 0.0, 0.0]))
    a = c.compute(s, np.array([1.0, 0.0, 0.0]), mass_kg=300.0)
    assert a.shape == (3,)
    assert isinstance(a[0], float) or hasattr(a[0], 'item')
