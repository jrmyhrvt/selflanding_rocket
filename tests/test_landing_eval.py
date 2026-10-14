import numpy as np
from state.core import State
from evaluation import landing as L


def _state(x, y, z, vx=0.0, vy=0.0, vz=0.0):
    s = State(pos=np.array([x, y, z]), vel=np.array([vx, vy, vz]))
    s.height_m = z
    return s


def test_no_data():
    res = L.evaluate_landing([], np.array([0.0, 0.0, 0.0]))
    assert res["status"] == L.NO_DATA


def test_landed_soft_and_on_target():
    hist = [_state(0.3, 0.0, 5.0, 0, 0, -5.0),
            _state(0.2, 0.0, 0.0, 0.1, 0.0, -0.4)]
    res = L.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]))
    assert res["status"] == L.LANDED
    assert res["contact"] is True


def test_too_fast_on_contact():
    hist = [_state(0.0, 0.0, 0.0, 0.0, 0.0, -3.0)]
    res = L.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]))
    assert res["status"] == L.TOO_FAST


def test_failed_when_still_airborne():
    hist = [_state(0.0, 0.0, 50.0, 0.0, 0.0, -1.0)]
    res = L.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]))
    assert res["status"] == L.FAILED
    assert res["contact"] is False


def test_failed_when_off_target():
    # Slow, touching ground, but far from the target point.
    hist = [_state(10.0, 0.0, 0.0, 0.0, 0.0, -0.2)]
    res = L.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]), dist_tol=1.0)
    assert res["status"] == L.FAILED


def test_ground_tolerance():
    # Slightly above ground but within h_tol -> counts as contact.
    hist = [_state(0.0, 0.0, 0.02, 0.0, 0.0, -0.1)]
    res = L.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]), h_tol=0.05)
    assert res["contact"] is True
    assert res["status"] == L.LANDED
