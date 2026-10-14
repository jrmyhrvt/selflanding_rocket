import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
import matplotlib.pyplot as plt

from visualization.viz3d_studio import LandingStudio
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
