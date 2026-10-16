"""Tests for the simplified VIRTUAL gimballed-thrust model.

The model is deliberately coarse; these tests only pin its documented contract
(direction, wrench = r x F, clamping, invertibility).
"""
import numpy as np

from physics.thrust_model import ThrustModel


def test_direction_is_unit_and_vertical_at_zero_gimbal():
    tm = ThrustModel()
    n = tm.thrust_direction_body(0.0, 0.0)
    assert np.allclose(n, [0.0, 0.0, 1.0])
    for gx, gy in [(0.2, 0.0), (0.0, -0.3), (0.15, 0.1)]:
        assert abs(np.linalg.norm(tm.thrust_direction_body(gx, gy)) - 1.0) < 1e-12


def test_force_scales_with_thrust():
    tm = ThrustModel(max_thrust_N=1000.0)
    F1, _, _ = tm.force_moment(100.0, 0.0, 0.0)
    F2, _, _ = tm.force_moment(500.0, 0.0, 0.0)
    assert np.isclose(np.linalg.norm(F2), 5.0 * np.linalg.norm(F1))


def test_thrust_and_gimbal_are_clamped():
    tm = ThrustModel(max_thrust_N=1000.0, gimbal_max_rad=0.2)
    F, _, _ = tm.force_moment(99999.0, 5.0, -5.0)
    assert np.isclose(np.linalg.norm(F), 1000.0)
    # negative thrust is clamped to zero
    Fz, _, _ = tm.force_moment(-10.0, 0.0, 0.0)
    assert np.allclose(Fz, 0.0)


def test_zero_thrust_gives_zero_force_and_moment():
    tm = ThrustModel()
    F, M, _ = tm.force_moment(0.0, 0.3, -0.3)
    assert np.allclose(F, 0.0)
    assert np.allclose(M, 0.0)


def test_moment_equals_r_cross_f():
    tm = ThrustModel(gimbal_offset_m=0.3)
    F, M, _ = tm.force_moment(1000.0, 0.12, -0.07)
    r = np.array([0.0, 0.0, -0.3])
    assert np.allclose(M, np.cross(r, F))
    # no roll authority: +Z moment is always zero
    assert abs(M[2]) < 1e-12


def test_gimbal_inverse_recovers_moment():
    tm = ThrustModel(max_thrust_N=1000.0, gimbal_offset_m=0.3)
    M_des = np.array([5.0, -3.0, 0.0])
    gx, gy = tm.gimbal_for_moment(M_des, 1000.0)
    _, M, _ = tm.force_moment(1000.0, gx, gy)
    assert np.allclose(M, M_des, atol=1e-9)


def test_gimbal_inverse_respects_limits():
    tm = ThrustModel(gimbal_max_rad=0.05, gimbal_offset_m=0.3)
    gx, gy = tm.gimbal_for_moment([1000.0, 1000.0, 0.0], 1000.0)
    assert abs(gx) <= 0.05 + 1e-12
    assert abs(gy) <= 0.05 + 1e-12


def test_commanded_moment_signs_map_to_gimbal_and_lateral_force():
    """The nozzle deflection must produce force AND moment in the right sense.

    Physics check for step 3: a +X body moment is produced by a +Y lateral
    force (M = r x F with r = -offset*z), which requires gx < 0. A +Y moment
    needs a -X force, hence gy < 0. If any sign were flipped, the attitude loop
    would drive the rocket the wrong way.
    """
    tm = ThrustModel(max_thrust_N=1000.0, gimbal_offset_m=0.3)
    T = 800.0

    gx, gy = tm.gimbal_for_moment([5.0, 0.0, 0.0], T)
    F_b, M_b, _ = tm.force_moment(T, gx, gy)
    assert M_b[0] > 0.0 and F_b[1] > 0.0 and gx < 0.0

    gx, gy = tm.gimbal_for_moment([0.0, 5.0, 0.0], T)
    F_b, M_b, _ = tm.force_moment(T, gx, gy)
    assert M_b[1] > 0.0 and F_b[0] < 0.0 and gy < 0.0
