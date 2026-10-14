import matplotlib
matplotlib.use("Agg")

import os
import csv
import pytest

from evaluation import experiment as E
from evaluation import landing as L


def _trial(sweep, value, status, success=None, dur=10.0, steps=100):
    return E.Trial(
        sweep=sweep, sweep_value=value, status=status,
        success=(status == L.LANDED) if success is None else success,
        contact=(status in (L.LANDED, L.TOO_FAST)),
        sim_duration_s=dur, end_height_m=0.0,
        v_vertical_m_s=-0.5, v_horizontal_m_s=0.0, v_total_m_s=0.5,
        target_distance_m=0.0, n_steps=steps,
        start_height_m=25.0, offset_x_m=0.0, offset_y_m=0.0,
        target_x_m=0.0, target_y_m=0.0,
        v0_vertical_m_s_init=0.0, v0_horizontal_m_s=0.0,
        dt_s=0.05, max_time_s=300.0,
    )


def test_summarize_counts_known_data():
    trials = [
        _trial("s", 1.0, L.LANDED),
        _trial("s", 2.0, L.LANDED),
        _trial("s", 3.0, L.TOO_FAST),
        _trial("s", 4.0, L.FAILED),
    ]
    summ = E.summarize(trials)
    assert summ["total"] == 4
    assert summ["landed"] == 2
    assert summ["failed"] == 2
    assert abs(summ["success_rate"] - 0.5) < 1e-12


def test_timeout_is_not_success():
    # A run that ended airborne (timeout) is FAILED -> never success.
    timeout = _trial("max_time_s", 2.0, L.FAILED)
    assert timeout.success is False
    assert E.summarize([timeout])["landed"] == 0


def test_group_by_sweep():
    trials = [_trial("a", 1.0, L.LANDED), _trial("a", 2.0, L.LANDED),
              _trial("b", 1.0, L.FAILED)]
    groups = E.group_by_sweep(trials)
    assert set(groups) == {"a", "b"}
    assert len(groups["a"]) == 2 and len(groups["b"]) == 1


def test_run_trial_reproducible():
    t1 = E.run_trial("start_height_m", 20.0)
    t2 = E.run_trial("start_height_m", 20.0)
    for field in ("status", "sim_duration_s", "end_height_m",
                  "v_vertical_m_s", "v_horizontal_m_s", "v_total_m_s",
                  "target_distance_m", "n_steps"):
        assert getattr(t1, field) == getattr(t2, field), field


def test_timeout_trial_measured_as_failed():
    # max_time too small -> airborne at end -> FAILED, not LANDED
    t = E.run_trial("max_time_s", 2.0)
    assert t.status == L.FAILED
    assert t.success is False
    assert t.contact is False
    assert t.end_height_m > L.H_TOL_M


def test_run_experiment_size_and_success_count():
    ranges = {"start_height_m": [5.0, 10.0], "dt": [0.1, 0.05]}
    trials = E.run_experiment(ranges)
    assert len(trials) == 4
    summ = E.summarize(trials)
    assert summ["total"] == 4
    assert summ["landed"] + summ["failed"] == 4


def test_csv_export(tmp_path):
    trials = [_trial("s", 1.0, L.LANDED), _trial("s", 2.0, L.FAILED)]
    path = str(tmp_path / "res.csv")
    out = E.write_results_csv(trials, path)
    assert os.path.exists(out)
    with open(out, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["status"] == L.LANDED
    assert "v_vertical_m_s" in rows[0]
    assert rows[1]["success"] in ("False", "0")


def test_plot_results_saves_file(tmp_path):
    trials = [_trial("start_height_m", 5.0, L.LANDED),
              _trial("start_height_m", 60.0, L.FAILED),
              _trial("dt", 0.1, L.LANDED)]
    path = str(tmp_path / "fig.png")
    E.plot_results(trials, path=path, show=False)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0


def test_unknown_sweep_raises():
    with pytest.raises(KeyError):
        E._scenario_for("gibts_nicht", 1.0)
