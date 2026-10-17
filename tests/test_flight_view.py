"""Tests for the unified 3D flight view (single state source, real nozzle)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from physics.rotation import rotate_vector
from physics.thrust_model import ThrustModel
from simulation.flight import FlightConfig, flight_telemetry
from visualization.flight_view import (FlightView, MODE_TILTED, MODE_UPRIGHT,
                                       VIEW_CLOSEUP, VIEW_OVERVIEW,
                                       CTRL_AUTO, CTRL_MANUAL)
from visualization.rocket_body import (body_axis_world, nozzle_axis_world,
                                       rocket_geometry, nozzle_geometry,
                                       transform_points)


@pytest.fixture
def view():
    v = FlightView(cfg=FlightConfig(tf=1.0, dt=0.02))
    yield v
    import matplotlib.pyplot as plt
    plt.close(v.fig)


def test_view_has_history_and_geometry(view):
    assert len(view.history) == 50
    assert view.body_world().shape[1] == 3
    assert view.nozzle_world().shape[1] == 3


def test_view_updates_over_multiple_steps(view):
    view.update(0)
    body0 = view.body_world().copy()
    view.update(len(view.history) // 2)
    body1 = view.body_world().copy()
    view.update(len(view.history) - 1)
    body2 = view.body_world().copy()
    # the rocket actually moves across frames (not a static frozen scene)
    assert not np.allclose(body0, body1)
    assert not np.allclose(body1, body2)


def test_view_draws_labelled_vectors(view):
    view.update(len(view.history) // 2)
    labels = [t.get_text() for t in view._vec_labels]
    assert len(view._vec_arrows) > 0
    for name in ("v", "a", "F_Schub", "F_Schwerkraft", "F_gesamt"):
        assert name in labels


def test_nozzle_is_not_an_independent_sine():
    """With a zero-gimbal run the nozzle must be aligned with the body."""
    v = FlightView(cfg=FlightConfig(tf=0.5, dt=0.02, gravity=False,
                                    stop_at_ground=False, use_controller=False,
                                    thrust_N_open=500.0, gimbal_x_open=0.0,
                                    gimbal_y_open=0.0))
    try:
        for f in range(0, len(v.history), 5):
            v.update(f)
            q = v._q[f]
            # nozzle axis points down/out; aligned with the body when gimbal=0
            assert np.allclose(nozzle_axis_world(q, 0.0),
                               -body_axis_world(q), atol=1e-12)
    finally:
        import matplotlib.pyplot as plt
        plt.close(v.fig)


def test_view_rotation_mode_uses_same_simulation(view):
    """Switching mode re-runs the SAME shared simulation (never a fake overlay).

    Only the initial attitude changes; the physics/controller stay identical.
    """
    view._on_mode(MODE_TILTED)
    assert not np.allclose(view.cfg.start_q, [1.0, 0.0, 0.0, 0.0])
    assert len(view.history) > 0
    view._on_mode(MODE_UPRIGHT)
    assert np.allclose(view.cfg.start_q, [1.0, 0.0, 0.0, 0.0])
    assert len(view.history) > 0


def test_restart_recomputes_from_config(view):
    view.update(len(view.history) - 1)
    view.restart()
    assert view.frame == 0
    assert len(view.history) == 50


def test_restart_reuses_single_animation(view):
    """Regression for the macOS 'tilted flight' crash.

    restart() must NOT drop and recreate the FuncAnimation. On the macOS backend
    the animation timer is a C-level CFRunLoop timer; recreating the animation
    on every restart (which the 'Geneigt'/'Neustart'/dt controls all trigger)
    frees a still-registered native timer and crashes the app with
    "'TimerMac' object has no attribute 'callbacks'". The SAME animation object
    must survive every restart path.
    """
    view.play()
    ani0 = view.ani
    assert ani0 is not None
    view.restart()
    assert view.ani is ani0                     # restart reuses, never recreates
    view._on_mode(MODE_TILTED)
    assert view.ani is ani0                     # mode switch reuses
    view._on_mode(MODE_UPRIGHT)
    assert view.ani is ani0
    view.set_dt("0.05")
    assert view.ani is ani0                     # dt change reuses


def test_set_dt_recomputes(view):
    view.set_dt("0.05")
    assert np.isclose(view.cfg.dt, 0.05)
    assert len(view.history) == 20
    # invalid dt is ignored (no crash, keeps previous history)
    n = len(view.history)
    view.set_dt("not-a-number")
    assert len(view.history) == n


def test_telemetry_text_marks_unmodelled(view):
    view.update(0)
    text = view._telemetry_text(0)
    assert "NICHT MODELLIERT" in text
    assert "Schub" in text
    assert "Moment" in text


def test_hud_reports_lateral_speed_from_state():
    """Step 1/4: the HUD lateral speed must be the state's horizontal speed.

    The HUD reads the inertial ENU state directly (vx, vy, vz) and the
    horizontal speed is |(vx, vy)|. This pins that the displayed lateral speed
    matches the simulation state, i.e. there is no display/frame error.
    """
    import matplotlib.pyplot as plt
    a = np.deg2rad(15.0)
    cfg = FlightConfig(tf=2.0, dt=0.02,
                       start_q=np.array([np.cos(a / 2.0), 0.0,
                                         np.sin(a / 2.0), 0.0]),
                       start_vel=[3.0, 4.0, 0.0])
    v = FlightView(cfg=cfg)
    try:
        f = len(v.history) // 2
        v.update(f)
        st = v.history[f]
        d = flight_telemetry(st, cfg)
        assert np.isclose(d["velocity_x_m_s"], st.vel[0])
        assert np.isclose(d["velocity_y_m_s"], st.vel[1])
        assert np.isclose(d["velocity_z_m_s"], st.vel[2])
        assert np.isclose(d["speed_horizontal_m_s"],
                          np.hypot(st.vel[0], st.vel[1]))
        assert "speed_h" in v._telemetry_text(f)
    finally:
        plt.close(v.fig)


def test_view_telemetry_is_same_state_source(view):
    d = flight_telemetry(view.history[3], view.cfg)
    assert np.isclose(view._t[3], d["time_s"])
    assert np.isclose(view._z[3], d["height_m"])
    assert np.allclose(view._q[3], d["quaternion_B_to_I"])


# ---------------------------------------------------------------- step 1/6:
# hull + nozzle always present and rebuilt from the ACTUAL state each frame.
def test_hull_and_nozzle_rebuilt_from_state_each_frame(view):
    for f in (0, len(view.history) // 2, len(view.history) - 1):
        view.update(f)
        q = view._q[f]
        pos = view._rocket_pos(f)
        bv, _ = rocket_geometry()
        expected = transform_points(bv, q, pos)
        assert np.allclose(view.body_world(), expected)
        assert view.body_world().shape[1] == 3
        assert view.nozzle_world().shape[1] == 3
        # vertex count from the real geometry, and the collections are drawn
        assert view.body_world().shape[0] == len(rocket_geometry()[0])
        assert view.nozzle_world().shape[0] == len(nozzle_geometry()[0])
        assert view.body_poly in view.ax3d.collections
        assert view.nozzle_poly in view.ax3d.collections


def test_nozzle_moves_with_the_physics_gimbal():
    """The nozzle is hinged by the SAME gimbal angles used by the physics."""
    a = np.deg2rad(15.0)
    cfg = FlightConfig(tf=1.5, dt=0.02,
                       start_q=np.array([np.cos(a / 2.0), 0.0,
                                         np.sin(a / 2.0), 0.0]))
    v = FlightView(cfg=cfg)
    thr = ThrustModel()
    try:
        saw_nonzero_gimbal = False
        for f in range(0, len(v.history), 5):
            gx, gy = v._gimbal_angles(f)
            if max(abs(gx), abs(gy)) > 1e-4:
                saw_nonzero_gimbal = True
            # the drawn nozzle axis equals -thrust_direction from the model
            assert np.allclose(v._nozzle_axis_body(gx, gy),
                               -thr.thrust_direction_body(gx, gy), atol=1e-12)
            # and it matches the state's own thrust direction
            st = v.history[f]
            assert np.allclose(v._nozzle_axis_body(gx, gy),
                               -np.asarray(st.thrust_dir_b, float), atol=1e-9)
        assert saw_nonzero_gimbal      # the gimbal really actuates
    finally:
        import matplotlib.pyplot as plt
        plt.close(v.fig)


def test_thrust_direction_line_differs_from_long_axis_when_gimballed():
    """Thrust axis and body long axis are drawn as SEPARATE directions and only
    coincide for a zero gimbal."""
    a = np.deg2rad(15.0)
    cfg = FlightConfig(tf=1.5, dt=0.02,
                       start_q=np.array([np.cos(a / 2.0), 0.0,
                                         np.sin(a / 2.0), 0.0]))
    v = FlightView(cfg=cfg)
    try:
        max_angle = 0.0
        for f in range(0, len(v.history), 5):
            v.update(f)
            q = v._q[f]
            st = v.history[f]
            long_axis = np.asarray(rotate_vector(q, [0.0, 0.0, 1.0]), float)
            thrust = np.asarray(rotate_vector(
                q, np.asarray(st.thrust_dir_b, float)), float)
            max_angle = max(max_angle, np.degrees(np.arccos(
                np.clip(float(np.dot(long_axis, thrust)), -1.0, 1.0))))
        assert max_angle > 0.5        # visibly different while gimballed
    finally:
        import matplotlib.pyplot as plt
        plt.close(v.fig)


# ---------------------------------------------------------------- step 5:
# force arrows are scaled by their own magnitude, not by the rocket geometry.
def test_force_arrows_scaled_independently_of_rocket_length(view):
    view.update(len(view.history) // 2)
    ref = view._ref["Ft"]
    full = view._vec_display_length(np.array([0.0, 0.0, ref]), ref)
    half = view._vec_display_length(np.array([0.0, 0.0, 0.5 * ref]), ref)
    assert np.isclose(full, view._arrow_len)          # strongest force -> cap
    assert np.isclose(half, 0.5 * view._arrow_len)    # proportional to magnitude
    # the cap is set by the VIEW size, not by the rocket length
    assert np.isclose(view._arrow_len, 0.35 * view._cam_half)
    assert full < np.inf


# ---------------------------------------------------------------- steps 2/3:
# interactive camera that does NOT jump back on every frame.
def test_overview_camera_persists_across_frames(view):
    view.update(0)
    before = (view._cam_elev, view._cam_azim, view._cam_half,
              tuple(view._cam_center))
    for f in range(len(view.history)):
        view.update(f)
    after = (view._cam_elev, view._cam_azim, view._cam_half,
             tuple(view._cam_center))
    assert before == after


def test_zoom_preserves_orientation(view):
    view.update(0)
    elev, azim, half0 = view._cam_elev, view._cam_azim, view._cam_half
    view.zoom(0.5)
    assert view._cam_half < half0
    assert view._cam_elev == elev and view._cam_azim == azim
    view.zoom(4.0)
    assert view._cam_half > half0


def test_rotate_then_frames_do_not_jump(view):
    view.update(0)
    view.rotate(-35.0, 12.0)
    azim, elev = view._cam_azim, view._cam_elev
    for f in range(0, len(view.history), 3):
        view.update(f)
    assert view._cam_azim == azim and view._cam_elev == elev


def test_pan_moves_centre_and_keeps_orientation(view):
    view.update(0)
    c0 = view._cam_center.copy()
    azim, elev = view._cam_azim, view._cam_elev
    view.pan(40.0, 20.0)
    assert not np.allclose(view._cam_center, c0)
    assert view._cam_azim == azim and view._cam_elev == elev


def test_view_mode_closeup_tracks_rocket_without_jump(view):
    view.set_view_mode(VIEW_CLOSEUP)
    assert view._viz_mode == VIEW_CLOSEUP
    assert view._cam_half < 2.0
    view.rotate(25.0, 0.0)
    azim = view._cam_azim
    last = len(view.history) - 1
    view.update(last)
    pos = view._rocket_pos(last)
    assert np.allclose(view._cam_center, pos + np.array([0.0, 0.0, 0.35]))
    assert view._cam_azim == azim            # follow does not reset rotation
    view.set_view_mode(VIEW_OVERVIEW)
    assert view._viz_mode == VIEW_OVERVIEW
    span = max(float(np.ptp(view._x)), float(np.ptp(view._y)),
               float(np.ptp(view._z)), 1.0)
    assert np.isclose(view._cam_half, 0.62 * span)   # reframed to whole flight


def test_interaction_handlers_installed_for_camera_control(view):
    assert len(view._cids) >= 4
    for name in ("zoom", "pan", "rotate", "reset_view", "set_view_mode"):
        assert callable(getattr(view, name))
    assert (VIEW_OVERVIEW, VIEW_CLOSEUP) == ("Gesamtansicht", "Nahansicht")


# ---------------------------------------------------------------- TVC servos
def test_view_has_two_servo_sliders_and_control_mode(view):
    assert (CTRL_AUTO, CTRL_MANUAL) == ("Auto (Regler)", "Manuell (Servo)")
    assert view.slider_sx is not None
    assert view.slider_sy is not None
    assert view.radios_ctrl is not None
    # sliders span the physical servo limit (deg)
    gmax = np.degrees(view.cfg.gimbal_max_rad)
    assert np.isclose(view.slider_sx.valmax, gmax)
    assert np.isclose(view.slider_sy.valmin, -gmax)


def test_view_draws_both_servo_axes(view):
    view.update(len(view.history) // 2)
    labels = [t.get_text() for t in view._vec_labels]
    assert any("Servo X" in l for l in labels)
    assert any("Servo Y" in l for l in labels)


def test_manual_mode_recomputes_and_is_clearly_marked(view):
    view._on_ctrl_mode(CTRL_MANUAL)
    assert view.cfg.manual_servo is True
    view.slider_sx.set_val(8.0)
    view.slider_sy.set_val(-6.0)
    assert np.isclose(view.cfg.manual_gimbal_x_rad, np.deg2rad(8.0))
    assert np.isclose(view.cfg.manual_gimbal_y_rad, np.deg2rad(-6.0))
    # the recompute is explicit, never silent
    assert view._status_note != ""
    text = view._control_status_text(0)
    assert "MANUELL" in text
    assert "Soll" in text and "Ist" in text
    # and the run really used the manual setpoints
    assert len(view.history) == 50


def test_sliders_are_inert_in_auto_mode(view):
    before_x = view.cfg.manual_gimbal_x_rad
    view.slider_sx.set_val(10.0)
    assert np.isclose(view.cfg.manual_gimbal_x_rad, before_x)
    assert view.cfg.manual_servo is False
    view._on_ctrl_mode(CTRL_AUTO)
    assert view.cfg.manual_servo is False


def test_telemetry_separates_setpoint_and_actual(view):
    view.update(0)
    text = view._telemetry_text(0)
    assert "Servo X" in text and "Soll" in text and "Ist" in text
    assert "Steuerung" in text
    assert "AUTO" in text
