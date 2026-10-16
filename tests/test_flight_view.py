"""Tests for the unified 3D flight view (single state source, real nozzle)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from simulation.flight import FlightConfig, flight_telemetry
from visualization.flight_view import FlightView, MODE_TILTED, MODE_UPRIGHT
from visualization.rocket_body import body_axis_world, nozzle_axis_world


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
