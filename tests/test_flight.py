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


def test_mass_and_fuel_burn_model_optional():
    # no isp -> constant mass
    hist = run_flight(FlightConfig(tf=1.0))
    assert np.isclose(hist[0].fuel_kg, hist[-1].fuel_kg)
    # with isp -> fuel decreases
    hist2 = run_flight(FlightConfig(tf=1.0, isp_s=250.0))
    assert hist2[-1].fuel_kg < hist2[0].fuel_kg
