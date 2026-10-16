import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from visualization.viz3d_studio import (
    LandingStudio, ROCKET_LENGTH_M, NOZZLE_LENGTH_M,
    GIMBAL_AMP_RAD, MODE_LANDING, MODE_ROTATION)
from state.core import State
from evaluation import landing as L


@pytest.fixture
def studio():
    st = LandingStudio()
    yield st
    plt.close(st.fig)


def test_starts_with_landed_run(studio):
    assert studio.result is not None
    assert studio.result["status"] == L.LANDED
    assert len(studio.history) > 1


def test_settings_boxes_exist(studio):
    for key in ("h0", "ox", "oy", "tx", "ty", "vz", "vx", "dt"):
        assert key in studio._boxes


def test_changing_settings_recomputes_new_trajectory(studio):
    old_hist = studio.history
    old_end = old_hist[-1].pos.copy()
    studio.set_setting("h0", "40")
    studio.set_setting("tx", "-5")
    studio.set_setting("ty", "0")
    res = studio.run_simulation()
    assert studio.history is not old_hist          # genuinely new run
    assert not np.allclose(studio.history[-1].pos, old_end)
    assert res["status"] == L.LANDED
    assert abs(studio.history[-1].pos[0] - (-5.0)) <= L.DIST_TOL_M


def test_initial_conditions_change_motion(studio):
    # td with a large downward initial velocity must reach the ground faster.
    studio.set_setting("h0", "15")
    studio.set_setting("vz", "0")
    slow = studio.run_simulation()
    n_slow = len(studio.history)

    studio.set_setting("vz", "-8")
    fast = studio.run_simulation()
    n_fast = len(studio.history)
    assert n_fast < n_slow


def test_readout_distinguishes_current_and_final(studio):
    studio.update(0)
    r0 = studio.readout()
    assert r0["frame"] == 0
    assert r0["live_status"] == "IN_FLIGHT"      # not yet landed mid-flight
    assert r0["final_status"] == L.LANDED        # whole-run result

    studio.update(len(studio.history) - 1)
    r1 = studio.readout()
    assert r1["live_status"] == L.LANDED
    assert r1["height"] <= L.H_TOL_M + 1e-9
    assert "LANDED" in r1["result_text"]
    assert "GESAMTERGEBNIS" in r1["result_text"]


def test_update_changes_displayed_data(studio):
    studio.update(0)
    xs0, _ = studio.trail.get_data()
    studio.update(len(studio.history) - 1)
    xs1, _ = studio.trail.get_data()
    assert len(xs1) > len(xs0)


def test_failed_landing_shown_as_failed_too_fast(studio):
    studio.set_setting("h0", "6")
    studio.set_setting("vz", "-30")   # arrives way too fast
    res = studio.run_simulation()
    assert res["status"] == L.TOO_FAST
    studio.update(len(studio.history) - 1)
    r = studio.readout()
    assert r["live_status"] == L.TOO_FAST
    assert r["final_status"] == L.TOO_FAST
    assert "TOO_FAST" in r["result_text"]


def test_failed_landing_shown_as_failed_timeout(studio):
    studio.max_time_s = 2.0
    studio.set_setting("h0", "60")
    res = studio.run_simulation()
    assert res["status"] == L.FAILED
    studio.update(len(studio.history) - 1)
    assert studio.readout()["live_status"] == L.FAILED


def test_reset_rewinds_without_recomputing(studio):
    studio.update(len(studio.history) - 1)
    hist = studio.history
    studio.reset_run()
    assert studio.frame == 0
    assert studio.history is hist          # reset does not recompute
    assert studio.paused is True


def test_pause_resume_toggles(studio):
    studio.play()
    assert studio.paused is False
    studio.toggle_pause()
    assert studio.paused is True
    studio.toggle_pause()
    assert studio.paused is False


def test_invalid_setting_reports_error(studio):
    studio.set_setting("dt", "0")
    res = studio.run_simulation()
    assert res is None
    assert "Ungueltige" in studio._message
    # old run kept intact
    assert len(studio.history) > 1


def test_unknown_setting_key_raises(studio):
    with pytest.raises(KeyError):
        studio.set_setting("nicht_vorhanden", "1")


def _load_history(studio, qs):
    """Replace the run with synthetic states carrying the given quaternions."""
    studio.history = [
        State(t=float(i), pos=np.array([i * 0.0, 0.0, 10.0 - i]),
              vel=np.zeros(3), height_m=10.0 - i, q=np.asarray(q, dtype=float))
        for i, q in enumerate(qs)
    ]
    studio._cache_arrays()
    return studio


def test_rocket_geometry_body_and_nozzle_are_separate(studio):
    assert studio.body_verts.shape[1] == 3
    assert studio.nozzle_verts.shape[1] == 3
    assert len(studio.body_faces) > 0
    assert len(studio.nozzle_faces) > 0
    # body is upward (+Z) with a defined tip length; nozzle hangs below z=0
    assert np.isclose(studio.body_verts[:, 2].max(), ROCKET_LENGTH_M)
    assert np.isclose(studio.nozzle_verts[:, 2].min(), -NOZZLE_LENGTH_M)


def test_body_orientation_identity_points_up(studio):
    _load_history(studio, [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    studio.update(0)
    assert np.allclose(studio.body_axis_world(), [0.0, 0.0, 1.0])
    pos = np.array([studio._x[0], studio._y[0], studio._z[0]])
    assert np.allclose(studio.rocket_tip_world(), pos + [0.0, 0.0, ROCKET_LENGTH_M])


def test_body_orientation_known_rotation(studio):
    s = np.sin(np.pi / 4.0)
    _load_history(studio, [[1.0, 0.0, 0.0, 0.0], [s, s, 0.0, 0.0]])
    studio.update(1)
    assert np.allclose(studio.body_axis_world(), [0.0, -1.0, 0.0], atol=1e-9)
    pos = np.array([studio._x[1], studio._y[1], studio._z[1]])
    assert np.allclose(studio.rocket_tip_world(),
                       pos + np.array([0.0, -ROCKET_LENGTH_M, 0.0]), atol=1e-9)


def test_body_orientation_tracks_current_frame(studio):
    q90x = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    _load_history(studio, [[1.0, 0.0, 0.0, 0.0], q90x])
    studio.update(0)
    assert np.allclose(studio.body_axis_world(), [0.0, 0.0, 1.0])
    studio.update(1)
    assert not np.allclose(studio.body_axis_world(), [0.0, 0.0, 1.0])


def test_nozzle_stays_attached_and_axis_points_down_without_gimbal(studio):
    _load_history(studio, [[1.0, 0.0, 0.0, 0.0]])
    studio.update(0)  # landing mode -> gimbal = 0
    assert np.isclose(studio.gimbal_angle, 0.0)
    pos = np.array([studio._x[0], studio._y[0], studio._z[0]])
    # no gimbal: nozzle bottom centre is exactly below the base along -axis
    bottom_center = studio.nozzle_world()[-1]
    axis = studio.body_axis_world()
    assert np.allclose(bottom_center, pos - axis * NOZZLE_LENGTH_M)
    # nozzle axis is opposite to body axis
    assert np.allclose(studio.nozzle_axis_world(), -axis)


def test_nozzle_relative_motion_follows_gimbal(studio):
    _load_history(studio, [[1.0, 0.0, 0.0, 0.0]])
    studio.update(0)
    # simulate a non-zero gimbal explicitly
    pos = np.array([studio._x[0], studio._y[0], studio._z[0]])
    theta = np.deg2rad(15.0)
    studio._update_rocket(np.array([1.0, 0.0, 0.0, 0.0]), pos, theta)
    axis = studio.body_axis_world()
    nax = studio.nozzle_axis_world()
    # angle between nozzle and -body axis equals the gimbal angle
    cosang = float(np.clip(np.dot(nax, -axis), -1.0, 1.0))
    assert np.isclose(np.arccos(cosang), theta, atol=1e-9)
    # nozzle moved relative to the (unchanged) body axis
    assert not np.allclose(studio.nozzle_world()[-1], pos - axis * NOZZLE_LENGTH_M)


def test_studio_adopts_external_history_via_public_entry():
    hist = [
        State(t=0.0, pos=np.array([0.0, 0.0, 10.0]), height_m=10.0,
              q=np.array([1.0, 0.0, 0.0, 0.0])),
        State(t=1.0, pos=np.array([0.0, 0.0, 9.0]), height_m=9.0,
              q=np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])),
    ]
    st = LandingStudio(history=hist)
    try:
        # adopted verbatim, not recomputed/overwritten by an auto-run
        assert len(st.history) == 2
        assert st.history[0] is hist[0]
        st.update(1)
        assert np.allclose(st.body_axis_world(), [0.0, -1.0, 0.0], atol=1e-9)
    finally:
        plt.close(st.fig)


def test_default_studio_still_autostarts(studio):
    assert len(studio.history) > 1
    assert studio.result is not None


# ------------------------------------------------------------ mode selector
def test_mode_selector_exists_and_defaults_to_landing(studio):
    from visualization.viz3d_studio import MODE_LANDING, MODE_ROTATION
    labels = [getattr(l, "get_text", lambda: l)() for l in studio.mode_radio.labels]
    assert set(labels) == {MODE_LANDING, MODE_ROTATION}
    assert studio.mode == MODE_LANDING
    # default studio still shows the landing run
    assert studio.result["status"] in (L.LANDED, L.TOO_FAST, L.FAILED)


def test_switch_to_rotation_mode_shows_rotating_rocket(studio):
    landing_hist = studio.history
    studio._on_mode_change(MODE_ROTATION)
    assert studio.mode == MODE_ROTATION
    assert studio.history is not landing_hist
    # a real rotation: quaternions change across the history
    q_first = np.asarray(studio.history[0].q)
    q_last = np.asarray(studio.history[-1].q)
    assert not np.allclose(q_first, q_last)
    # body axis is tilted at the final frame (not simply +Z)
    studio.update(len(studio.history) - 1)
    assert not np.allclose(studio.body_axis_world(), [0.0, 0.0, 1.0])


def test_rotation_mode_shows_gimbal_demo(studio):
    studio._on_mode_change(MODE_ROTATION)
    # gimbal oscillates over time -> values differ between frames
    n = len(studio.history)
    angles = []
    for f in (0, n // 4, n // 2, 3 * n // 4):
        studio.update(f)
        angles.append(studio.gimbal_angle)
    assert any(abs(a) > 1e-6 for a in angles)
    assert not np.allclose(angles, angles[0])   # not constant
    # demo amplitude is bounded
    assert all(abs(a) <= GIMBAL_AMP_RAD + 1e-9 for a in angles)


def test_landing_mode_keeps_gimbal_at_zero(studio):
    studio._on_mode_change(MODE_LANDING)
    for f in (0, len(studio.history) // 2, len(studio.history) - 1):
        studio.update(f)
        assert np.isclose(studio.gimbal_angle, 0.0)


def test_rotation_mode_frames_the_rocket_closely(studio):
    studio._on_mode_change(MODE_ROTATION)
    xspan = studio.ax3d.get_xlim()[1] - studio.ax3d.get_xlim()[0]
    # zoomed in: much smaller than the landing trajectory extent
    assert xspan < 5.0


def test_switch_back_to_landing_restores_landing_run(studio):
    from visualization.viz3d_studio import MODE_LANDING, MODE_ROTATION
    studio._on_mode_change(MODE_ROTATION)
    assert studio.mode == MODE_ROTATION
    studio._on_mode_change(MODE_LANDING)
    assert studio.mode == MODE_LANDING
    # landing run works again and has a real (non-rotating-tilt) result
    assert studio.result is not None
    assert studio.history[-1].t > 0.0


def test_landing_mode_does_not_integrate_rotation(studio):
    # standard landing history has identity quaternions (rotation opt-in only)
    assert all(np.allclose(s.q, [1.0, 0.0, 0.0, 0.0]) for s in studio.history)
