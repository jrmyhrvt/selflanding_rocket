"""Tests for the virtual TVC servo model (two independent single-axis servos).

Pins the documented contract: setpoint clamping, angle limits, slew-rate limit,
first-order response, no single-step jump, two independent axes.
"""
import numpy as np

from physics.servo import Servo, ServoModel


def test_servo_starts_at_zero_and_at_target():
    s = Servo(angle_max_rad=0.35, rate_max_rad_s=2.0, tau_s=0.05)
    assert s.angle == 0.0
    assert s.step(0.0, 0.01) == 0.0


def test_servo_respects_angle_limit_even_for_huge_setpoint():
    s = Servo(angle_max_rad=0.2, rate_max_rad_s=100.0, tau_s=0.0)
    for _ in range(1000):
        a = s.step(5.0, 0.01)
    assert abs(a) <= 0.2 + 1e-12
    assert abs(a - 0.2) < 1e-9                  # reached the physical limit
    assert s.saturated                           # commanded beyond the limit


def test_servo_never_jumps_more_than_rate_limit():
    s = Servo(angle_max_rad=1.0, rate_max_rad_s=0.5, tau_s=1e-9)
    dt = 0.01
    prev = s.angle
    for _ in range(50):
        a = s.step(1.0, dt)
        assert abs(a - prev) <= 0.5 * dt + 1e-9  # slew-rate limited
        prev = a


def test_servo_first_order_approaches_without_overshoot():
    s = Servo(angle_max_rad=1.0, rate_max_rad_s=10.0, tau_s=0.1)
    target = 0.4
    last = s.angle
    for _ in range(2000):
        last = s.step(target, 0.005)
    assert abs(last - target) < 1e-3
    # monotonic approach, never overshoots the setpoint
    s2 = Servo(angle_max_rad=1.0, rate_max_rad_s=10.0, tau_s=0.1)
    for _ in range(2000):
        a = s2.step(target, 0.005)
        assert a <= target + 1e-9


def test_servo_reported_rate_matches_applied_change():
    s = Servo(angle_max_rad=1.0, rate_max_rad_s=2.0, tau_s=0.05)
    dt = 0.01
    for _ in range(10):
        a0 = s.angle
        a1 = s.step(0.3, dt)
        assert np.isclose((a1 - a0) / dt, s.rate, atol=1e-9)


def test_servo_negative_dt_is_safe():
    s = Servo(angle_max_rad=0.35, rate_max_rad_s=2.0, tau_s=0.05)
    assert s.step(0.1, 0.0) == s.angle
    assert s.step(0.1, -0.01) == s.angle


def test_servo_reset():
    s = Servo(angle_max_rad=0.35)
    s.step(0.2, 0.1)
    s.reset(0.1)
    assert np.isclose(s.angle, 0.1)
    assert s.rate == 0.0


def test_servo_model_two_axes_are_independent():
    m = ServoModel(angle_max_rad=0.5, rate_max_rad_s=100.0, tau_s=0.0)
    ax, ay = m.step(0.2, -0.3, 0.01)
    assert np.isclose(ax, 0.2)
    assert np.isclose(ay, -0.3)
    assert np.isclose(m.angle_x, 0.2)
    assert np.isclose(m.angle_y, -0.3)
    # moving only X leaves Y unchanged
    m.reset(0.0, 0.0)
    m.step(0.4, 0.0, 0.01)
    assert np.isclose(m.angle_y, 0.0)
    assert np.isclose(m.angle_x, 0.4)


def test_servo_signs_follow_setpoint_sign():
    m = ServoModel(angle_max_rad=0.5, rate_max_rad_s=100.0, tau_s=0.0)
    ax, ay = m.step(-0.2, 0.3, 0.01)
    assert ax < 0.0 and ay > 0.0


def test_servo_invalid_parameters_rejected():
    import pytest
    with pytest.raises(ValueError):
        Servo(angle_max_rad=-1.0)
    with pytest.raises(ValueError):
        Servo(rate_max_rad_s=-1.0)
    with pytest.raises(ValueError):
        Servo(tau_s=-1.0)
