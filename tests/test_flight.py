"""Tests for the unified flight simulation (translation + rotation coupled).

Learning model only. These tests pin the documented physical contract:
- forces change ONLY translational state,
- moments change ONLY rotational state,
- orientation COUPLES the two (thrust direction depends on q),
- quaternions stay normalized,
- zero moment + zero initial rate => NO artificial rotation.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np

from physics.rotation import integrate_rotation_body, norm as qnorm
from physics.rigidbody import RigidBody
from physics.translation import Translation
from state.core import State
from simulation.flight import FlightConfig, flight_telemetry, run_flight
from visualization.rocket_body import body_axis_world


def _tilt_q(deg, axis=(0.0, 1.0, 0.0)):
    a = np.deg2rad(deg)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    return np.array([np.cos(a / 2.0), *(np.sin(a / 2.0) * ax)])


def test_run_flight_produces_history():
    hist = run_flight(FlightConfig(tf=1.0, dt=0.01, stop_at_ground=False))
    assert len(hist) == 100
    assert hist[-1].t > hist[0].t


def test_quaternions_stay_normalized_everywhere():
    hist = run_flight(FlightConfig(tf=5.0, start_q=_tilt_q(20.0)))
    assert all(abs(qnorm(s.q) - 1.0) < 1e-9 for s in hist)


def test_gravity_only_does_not_rotate():
    """Zero thrust => zero moment => no artificial rotation, but it falls."""
    hist = run_flight(FlightConfig(tf=2.0, start_pos=np.array([0.0, 0.0, 20.0]),
                                   use_controller=False, thrust_N_open=0.0))
    assert np.allclose(hist[-1].q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(hist[-1].omega_b, 0.0)
    assert hist[-1].pos[2] < hist[0].pos[2]          # translation changed
    assert np.allclose(hist[-1].pos[:2], 0.0)        # no horizontal motion


def test_thrust_without_gimbal_changes_only_translation():
    """Force through the COM => no moment => rotation stays identity."""
    hist = run_flight(FlightConfig(tf=1.0, gravity=False, stop_at_ground=False,
                                   use_controller=False, thrust_N_open=1000.0,
                                   gimbal_x_open=0.0, gimbal_y_open=0.0))
    assert np.allclose(hist[-1].q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(hist[-1].omega_b, 0.0)
    assert hist[-1].vel[2] > 0.0                     # accelerated upward


def test_gimballed_thrust_produces_rotation():
    ref = run_flight(FlightConfig(tf=1.0, gravity=False, stop_at_ground=False,
                                  use_controller=False, thrust_N_open=1000.0,
                                  gimbal_x_open=0.0, gimbal_y_open=0.0))
    turned = run_flight(FlightConfig(tf=1.0, gravity=False, stop_at_ground=False,
                                     use_controller=False, thrust_N_open=1000.0,
                                     gimbal_x_open=0.1, gimbal_y_open=0.0))
    assert np.allclose(ref[-1].q, [1.0, 0.0, 0.0, 0.0])
    assert not np.allclose(turned[-1].q, [1.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(turned[-1].omega_b) > 0.0
    assert np.linalg.norm(turned[-1].moment_total) > 0.0


def test_force_affects_only_translation_lowlevel():
    """Directly: a pure force updates pos/vel/acc and leaves q/omega_b alone."""
    s = State(pos=np.zeros(3), vel=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    s.omega_b = np.array([0.3, -0.2, 0.1])
    q_before, w_before = s.q.copy(), s.omega_b.copy()
    Translation(0.01).step(s, np.array([10.0, 0.0, -5.0]), mass_kg=1.0)
    assert not np.allclose(s.vel, 0.0)
    assert np.allclose(s.q, q_before)
    assert np.allclose(s.omega_b, w_before)


def test_moment_affects_only_rotation_lowlevel():
    """Directly: a pure moment updates q/omega_b and leaves pos/vel alone."""
    s = State(pos=np.array([1.0, 2.0, 3.0]), vel=np.array([0.5, 0.0, 0.0]),
              q=np.array([1.0, 0.0, 0.0, 0.0]))
    s.moment_total = np.array([1.0, 0.0, 0.0])
    pos_before, vel_before = s.pos.copy(), s.vel.copy()
    rb = RigidBody(mass_kg=10.0, I_diag=[1.0, 2.0, 3.0])
    integrate_rotation_body(s, 0.01, rb=rb)
    assert not np.allclose(s.omega_b, 0.0)
    assert np.allclose(s.pos, pos_before)
    assert np.allclose(s.vel, vel_before)


def test_attitude_couples_translation():
    """Tilted start: attitude recovers AND the thrust vector bends the path."""
    hist = run_flight(FlightConfig(tf=30.0, start_q=_tilt_q(15.0),
                                   start_vel=np.zeros(3)))
    tilt0 = np.degrees(np.arccos(np.clip(body_axis_world(hist[0].q)[2], -1, 1)))
    tiltT = np.degrees(np.arccos(np.clip(body_axis_world(hist[-1].q)[2], -1, 1)))
    assert abs(tilt0 - 15.0) < 1.0
    assert tiltT < 1.0                                # attitude recovered
    assert abs(hist[-1].pos[0]) > 1.0                 # lateral motion -> coupling


def test_controller_keeps_rotation_bounded():
    """The fix for the old runaway spin: gimbal is a real actuator, no blow-up."""
    hist = run_flight(FlightConfig(tf=30.0, start_q=_tilt_q(15.0)))
    max_rate = max(np.linalg.norm(s.omega_b) for s in hist)
    assert max_rate < 5.0                             # bounded, not spinning up
    assert abs(hist[-1].omega_b[1]) < 1e-2            # rate damped to zero


def test_gimballed_thrust_torques_in_commanded_direction():
    """A +X body moment must rotate the body about +X (q_x grows), not -X."""
    s = State(pos=np.zeros(3), vel=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    s.moment_total = np.array([2.0, 0.0, 0.0])
    rb = RigidBody(mass_kg=10.0, I_diag=[1.0, 1.0, 1.0])
    integrate_rotation_body(s, 0.01, rb=rb)
    assert s.alpha_b[0] > 0.0
    assert s.omega_b[0] > 0.0
    assert s.q[1] > 0.0                                # rotation about +X


def test_tilted_start_with_horizontal_velocity_is_damped():
    """Step 4: tilted start + initial lateral speed -> the controller removes it.

    The controller must actually USE the horizontal velocity (not just correct
    attitude): with the lateral loop the horizontal speed is driven to ~0 and
    the drift is far smaller than with attitude-only control (kv_vh = 0).
    """
    cfg = FlightConfig(tf=20.0, dt=0.01, start_q=_tilt_q(15.0),
                       start_vel=[5.0, 0.0, 0.0])
    hist = run_flight(cfg)
    assert abs(hist[0].vel[0] - 5.0) < 0.2             # initial condition present
    assert abs(hist[-1].vel[0]) < 0.2                  # lateral velocity removed

    legacy = run_flight(FlightConfig(tf=20.0, dt=0.01, start_q=_tilt_q(15.0),
                                     start_vel=[5.0, 0.0, 0.0], kv_vh=0.0))
    assert abs(legacy[-1].vel[0]) > 1.0               # attitude-only: not removed
    assert abs(hist[-1].pos[0]) < abs(legacy[-1].pos[0])
    max_rate = max(np.linalg.norm(s.omega_b) for s in hist)
    assert max_rate < 5.0                              # still bounded/stable


def test_lateral_damping_is_sign_independent():
    """Opposite lateral velocities are both reduced, not amplified."""
    for vx0 in (5.0, -5.0):
        hist = run_flight(FlightConfig(tf=20.0, dt=0.01, start_q=_tilt_q(15.0),
                                       start_vel=[vx0, 0.0, 0.0]))
        assert abs(hist[-1].vel[0]) < 0.2


def test_lateral_damping_on_arbitrary_axis():
    """Lateral damping works for a tilt and velocity on an arbitrary horizontal axis."""
    hist = run_flight(FlightConfig(tf=25.0, dt=0.01, start_q=_tilt_q(12.0, (1.0, 0.0, 0.0)),
                                   start_vel=[3.0, -2.0, 0.0]))
    assert np.hypot(hist[-1].vel[0], hist[-1].vel[1]) < 0.3


def test_tilted_flight_stays_finite_over_many_steps():
    """Regression: the reported 'tilted flight' crash.

    The tilted start is the only run that produces a nonzero attitude error and
    therefore a nonzero moment (the upright run has zero error -> zero moment).
    Every state value must stay finite (no NaN/inf) over the whole run and the
    quaternion must stay normalized.
    """
    hist = run_flight(FlightConfig(tf=20.0, dt=0.01, start_q=_tilt_q(15.0)))
    assert len(hist) > 100
    fields = ("pos", "vel", "acc", "q", "omega_b", "alpha_b",
              "force_thrust", "force_gravity", "force_total",
              "thrust_dir_b", "moment_total")
    for s in hist:
        for name in fields:
            assert np.all(np.isfinite(getattr(s, name))), (s.t, name)
        assert abs(qnorm(s.q) - 1.0) < 1e-9


def test_telemetry_is_consistent_with_state():
    hist = run_flight(FlightConfig(tf=1.0))
    st = hist[len(hist) // 2]
    d = flight_telemetry(st, FlightConfig(tf=1.0))
    assert np.isclose(d["total_mass_kg"], st.mass_kg)
    assert np.allclose(d["quaternion_B_to_I"], st.q)
    assert np.allclose(d["omega_b_rad_s"], st.omega_b)
    assert np.allclose(d["force_thrust_N"], st.force_thrust)
    assert np.allclose(d["moment_total_body_Nm"], st.moment_total)
    # resultant = thrust + gravity (inertial)
    assert np.allclose(d["force_total_N"], st.force_thrust + st.force_gravity)
    assert np.isclose(d["force_total_mag_N"],
                      np.linalg.norm(st.force_thrust + st.force_gravity))


def test_telemetry_labels_unmodelled_quantities():
    hist = run_flight(FlightConfig(tf=0.5))
    d = flight_telemetry(hist[0], FlightConfig(tf=0.5))
    assert d["drag_modeled"] is False
    assert d["aerodynamics_modeled"] is False
    assert d["modeled_thrust"] is True
    assert "modell" in d["thrust_model_note"].lower()


def test_moment_recorded_matches_thrust_geometry():
    cfg = FlightConfig(tf=0.5, gravity=False, stop_at_ground=False,
                       use_controller=False, thrust_N_open=800.0,
                       gimbal_x_open=0.08, gimbal_y_open=-0.04,
                       gimbal_offset_m=0.3)
    hist = run_flight(cfg)
    st = hist[2]
    r = np.array([0.0, 0.0, -0.3])
    F_b = st.thrust_N * np.asarray(st.thrust_dir_b, float)
    assert np.allclose(st.moment_total, np.cross(r, F_b))


# ---------------------------------------------------------------- TVC servos
def test_servo_setpoint_and_actual_are_separated():
    """The controller command is only a SETPOINT; the actual angle lags it."""
    cfg = FlightConfig(tf=1.0, dt=0.01, start_q=_tilt_q(15.0),
                       servo_tau_s=0.2, servo_rate_max_rad_s=0.5)
    hist = run_flight(cfg)
    # early in the run the servo has not reached the commanded setpoint yet
    st = hist[2]
    lag = max(abs(st.gimbal_x - st.gimbal_x_cmd),
              abs(st.gimbal_y - st.gimbal_y_cmd))
    assert lag > 1e-4
    # and the setpoint itself is a genuine command (tilt produces a command)
    assert max(max(abs(s.gimbal_x_cmd), abs(s.gimbal_y_cmd)) for s in hist) > 1e-3


def test_servo_actual_angle_never_exceeds_limit():
    cfg = FlightConfig(tf=2.0, dt=0.01, use_controller=False,
                       thrust_N_open=1000.0, gimbal_x_open=10.0,
                       gimbal_y_open=-10.0, gimbal_max_rad=0.2)
    hist = run_flight(cfg)
    for s in hist:
        assert abs(s.gimbal_x) <= 0.2 + 1e-12
        assert abs(s.gimbal_y) <= 0.2 + 1e-12


def test_controller_respects_servo_limits():
    """The controller never commands/achieves more than the servo limit."""
    cfg = FlightConfig(tf=5.0, dt=0.01, start_q=_tilt_q(15.0),
                       start_vel=[8.0, -4.0, 0.0], gimbal_max_rad=0.1)
    hist = run_flight(cfg)
    for s in hist:
        assert abs(s.gimbal_x) <= 0.1 + 1e-9
        assert abs(s.gimbal_y) <= 0.1 + 1e-9
        assert abs(s.gimbal_x_cmd) <= 0.1 + 1e-9
        assert abs(s.gimbal_y_cmd) <= 0.1 + 1e-9


def test_servo_rate_is_bounded_by_config():
    dt = 0.01
    cfg = FlightConfig(tf=1.0, dt=dt, use_controller=False, thrust_N_open=500.0,
                       gimbal_x_open=0.3, servo_rate_max_rad_s=0.7)
    hist = run_flight(cfg)
    for s in hist:
        assert abs(s.servo_rate_x) <= 0.7 + 1e-9
        assert abs(s.servo_rate_y) <= 0.7 + 1e-9


def test_manual_servo_x_moves_thrust_toward_negative_y():
    """Servo X on the body: +gimbal_x tilts the thrust towards -F_b_y."""
    cfg = FlightConfig(tf=0.5, dt=0.01, gravity=False, stop_at_ground=False,
                       manual_servo=True, manual_thrust_N=1000.0,
                       manual_gimbal_x_rad=0.3, manual_gimbal_y_rad=0.0,
                       servo_rate_max_rad_s=100.0, servo_tau_s=0.0)
    hist = run_flight(cfg)
    st = hist[-1]
    assert st.gimbal_x > 0.0
    assert st.thrust_dir_b[1] < 0.0            # body thrust leans to -Y
    assert st.moment_total[0] < 0.0            # M = r x F sign follows


def test_manual_servo_y_moves_thrust_toward_positive_x():
    cfg = FlightConfig(tf=0.5, dt=0.01, gravity=False, stop_at_ground=False,
                       manual_servo=True, manual_thrust_N=1000.0,
                       manual_gimbal_x_rad=0.0, manual_gimbal_y_rad=0.3,
                       servo_rate_max_rad_s=100.0, servo_tau_s=0.0)
    hist = run_flight(cfg)
    st = hist[-1]
    assert st.gimbal_y > 0.0
    assert st.thrust_dir_b[0] > 0.0            # body thrust leans to +X


def test_manual_servo_overrides_controller_and_keeps_hover_thrust():
    m_g = (50.0 + 10.0) * 9.80665
    base = FlightConfig(tf=0.2, dt=0.01, manual_servo=True,
                        manual_thrust_N=0.0, manual_gimbal_x_rad=0.0,
                        manual_gimbal_y_rad=0.0, use_controller=True)
    hist = run_flight(base)
    assert np.isclose(hist[0].thrust_N, m_g, rtol=1e-6)   # hover thrust
    # the controller would tilt; manual keeps the setpoint at zero
    assert np.allclose([s.gimbal_x_cmd for s in hist], 0.0, atol=1e-9)


def test_servo_couples_force_and_moment_to_motion():
    """The actual servo angle reaches the force AND the moment of the state."""
    cfg = FlightConfig(tf=0.5, dt=0.01, gravity=False, stop_at_ground=False,
                       manual_servo=True, manual_thrust_N=1000.0,
                       manual_gimbal_x_rad=0.2, servo_tau_s=0.0,
                       servo_rate_max_rad_s=100.0)
    hist = run_flight(cfg)
    st = hist[-1]
    r = np.array([0.0, 0.0, -0.3])
    F_b = st.thrust_N * np.asarray(st.thrust_dir_b, float)
    assert np.allclose(st.moment_total, np.cross(r, F_b), atol=1e-9)
    assert np.linalg.norm(st.omega_b) > 0.0            # moment rotated the body


def test_telemetry_exposes_servo_setpoint_and_actual():
    hist = run_flight(FlightConfig(tf=0.5, start_q=_tilt_q(15.0)))
    d = flight_telemetry(hist[5], FlightConfig(tf=0.5))
    for key in ("gimbal_x_rad", "gimbal_y_rad", "gimbal_x_cmd_rad",
                "gimbal_y_cmd_rad", "servo_rate_x_rad_s", "servo_rate_y_rad_s",
                "servo_x_saturated", "servo_y_saturated", "servo_manual"):
        assert key in d
    assert d["servo_manual"] is False


def test_mass_and_fuel_burn_model_optional():
    # no isp -> constant mass
    hist = run_flight(FlightConfig(tf=1.0))
    assert np.isclose(hist[0].fuel_kg, hist[-1].fuel_kg)
    # with isp -> fuel decreases
    hist2 = run_flight(FlightConfig(tf=1.0, isp_s=250.0))
    assert hist2[-1].fuel_kg < hist2[0].fuel_kg


# ---------------------------------------------------------------- position loop
def _offset_cfg(**kw):
    base = dict(tf=40.0, dt=0.02, start_pos=np.array([2.0, 0.0, 15.0]),
                start_vel=np.zeros(3), vz_des_m_s=-0.4,
                target_pos=np.array([0.0, 0.0, 0.0]))
    base.update(kw)
    return FlightConfig(**base)


def test_position_loop_disabled_by_default_is_legacy():
    """kp_xy = 0 (default) keeps the velocity-only behaviour EXACTLY."""
    a = run_flight(FlightConfig(tf=10.0, dt=0.02,
                                start_pos=np.array([2.0, 0.0, 15.0]),
                                start_vel=np.zeros(3), vz_des_m_s=-0.4))
    b = run_flight(FlightConfig(tf=10.0, dt=0.02,
                                start_pos=np.array([2.0, 0.0, 15.0]),
                                start_vel=np.zeros(3), vz_des_m_s=-0.4,
                                target_pos=np.array([0.0, 0.0, 0.0]),
                                kp_xy=0.0))
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.allclose(sa.pos, sb.pos)
        assert np.allclose(sa.vel, sb.vel)
        assert np.allclose(sa.gimbal_x, sb.gimbal_x)


def test_position_loop_removes_a_horizontal_offset():
    """With a target and kp_xy > 0 the vehicle is driven back over the target."""
    legacy = run_flight(_offset_cfg(kp_xy=0.0))
    ctrl = run_flight(_offset_cfg(kp_xy=0.4))
    assert abs(legacy[-1].pos[0]) > 1.0                # offset only damped
    assert abs(ctrl[-1].pos[0]) < 0.2                  # driven back to target
    assert abs(ctrl[-1].pos[0]) < abs(legacy[-1].pos[0])


def test_position_loop_sign_is_correct_on_both_axes():
    for x0, y0 in ((2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)):
        h = run_flight(_offset_cfg(start_pos=np.array([x0, y0, 15.0]),
                                   kp_xy=0.4))
        assert np.hypot(h[-1].pos[0], h[-1].pos[1]) < 0.2, (x0, y0)


def test_horizontal_velocity_command_is_clamped_and_target_ward():
    from simulation.flight import _horizontal_velocity_command
    s = State(pos=np.array([100.0, -100.0, 5.0]), vel=np.zeros(3),
              q=np.array([1.0, 0.0, 0.0, 0.0]))
    cfg = FlightConfig(target_pos=np.zeros(3), kp_xy=1.0, vh_max_m_s=2.0)
    vh = _horizontal_velocity_command(s, cfg)
    assert np.linalg.norm(vh) <= 2.0 + 1e-9            # clamped
    assert vh[0] < 0.0 and vh[1] > 0.0                 # points toward target
    # kp_xy = 0 -> plain feed-forward
    assert np.allclose(_horizontal_velocity_command(
        s, FlightConfig(target_pos=np.zeros(3), kp_xy=0.0)), 0.0)


def test_position_loop_does_not_add_horizontal_speed_at_touchdown():
    h = run_flight(_offset_cfg(kp_xy=0.4))
    assert np.hypot(h[-1].vel[0], h[-1].vel[1]) < 0.2


def test_position_loop_with_noisy_observation_still_converges():
    from sensors.observed import SensorConfig
    from sensors.observed_flight import ObservedFlightController
    from simulation.flight import model_control
    ctrl = ObservedFlightController(
        model_control, SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, seed=1234))
    h = run_flight(_offset_cfg(kp_xy=0.4), controller=ctrl)
    assert np.hypot(h[-1].pos[0], h[-1].pos[1]) < 0.6
