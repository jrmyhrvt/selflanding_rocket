from dataclasses import asdict
import numpy as np
import pytest

from simulation.scenario import Scenario
from evaluation import landing as L
from evaluation import sensor_experiment as SE
from evaluation import sensor_diagnostics as D


FAIL_SEED = 1234
SCENARIOS = [
    ("h20_d0", Scenario(start_height_m=20.0, target_xy=(0.0, 0.0))),
    ("h30_d2y", Scenario(start_height_m=30.0, target_xy=(2.0, 1.0),
                         v0_vertical_m_s=-2.0)),
]


@pytest.fixture(scope="module")
def failed_trial_series():
    sc = SCENARIOS[1][1]
    cfg = SE.replace(SE.DEFAULT_SENSOR_CFG, seed=FAIL_SEED)
    return SE.run_group_trial_with_series("estimation", "h30_d2y", sc, cfg)


def _fld_equal(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return a == b


def test_estimation_failure_is_too_fast_touchdown(failed_trial_series):
    trial, series = failed_trial_series
    assert trial.status == L.TOO_FAST
    assert trial.contact is True
    assert trial.end_speed_m_s > L.V_TOL_M_S


def test_find_failures_filters_mode_and_status():
    trials = SE.run_repeated(scenarios=SCENARIOS, seeds=(FAIL_SEED,),
                             groups=("estimation",))
    fails = D.find_failures(trials)
    assert len(fails) == 1
    assert fails[0].scenario == "h30_d2y"
    assert fails[0].group == "estimation"
    assert all(f.status in (L.TOO_FAST, L.FAILED) for f in fails)


def test_recording_does_not_change_result():
    sc = SCENARIOS[1][1]
    cfg = SE.replace(SE.DEFAULT_SENSOR_CFG, seed=FAIL_SEED)
    a = SE.run_group_trial("estimation", "h30_d2y", sc, cfg)
    b, _series = SE.run_group_trial_with_series("estimation", "h30_d2y", sc, cfg)
    da, db = asdict(a), asdict(b)
    for k in da:
        assert _fld_equal(da[k], db[k]), k


def test_series_comes_from_real_run(failed_trial_series):
    _, series = failed_trial_series
    assert len(series) >= 1
    keys = {"t", "pos_true", "vel_true", "pos_meas", "vel_meas",
            "pos_est", "vel_est", "pos_valid", "vel_valid"}
    assert set(series[0].keys()) == keys
    assert series[0]["t"] == 0.0
    for row in series:
        assert np.all(np.isfinite(row["vel_true"]))
        assert np.all(np.isfinite(row["vel_est"]))
        assert isinstance(row["vel_valid"], bool)


def test_series_length_matches_steps(failed_trial_series):
    trial, series = failed_trial_series
    assert len(series) == trial.n_steps


def test_estimation_series_has_dropouts_and_estimate(failed_trial_series):
    trial, series = failed_trial_series
    n_dropped = sum(1 for r in series if not r["vel_valid"])
    assert trial.missing_samples > 0
    assert n_dropped >= 1
    # estimate should differ from the raw measurement somewhere
    diffs = [np.linalg.norm(r["vel_est"] - r["vel_meas"])
             for r in series if r["vel_meas"] is not None]
    assert any(d > 1e-9 for d in diffs)


def test_noise_mode_has_no_dropped_meas_and_estimate_tracks_truth():
    sc = SCENARIOS[0][1]
    cfg = SE.replace(SE.DEFAULT_SENSOR_CFG, seed=FAIL_SEED)
    trial, series = SE.run_group_trial_with_series("noise", "h20_d0", sc, cfg)
    assert all(r["vel_valid"] for r in series)
    # no estimator -> the controller uses the (held) measurement directly
    for r in series:
        diff = np.linalg.norm(r["vel_est"] - r["vel_meas"])
        assert diff < 1e-9


def test_lag_metrics_consistent_with_series(failed_trial_series):
    trial, series = failed_trial_series
    m = D.lag_metrics(trial, series)
    assert m["t_contact_s"] == series[-1]["t"]
    assert m["vz_true_end_m_s"] == series[-1]["vel_true"][2]
    assert m["vz_est_end_m_s"] == series[-1]["vel_est"][2]
    assert abs(m["vz_end_delta_m_s"]
               - (m["vz_est_end_m_s"] - m["vz_true_end_m_s"])) < 1e-12
    dev = np.array([r["vel_est"][2] - r["vel_true"][2] for r in series])
    assert abs(m["max_abs_dev_m_s"] - np.max(np.abs(dev))) < 1e-12


def test_reproducible_series_same_seed():
    sc = SCENARIOS[1][1]
    cfg = SE.replace(SE.DEFAULT_SENSOR_CFG, seed=FAIL_SEED)
    _, a = SE.run_group_trial_with_series("estimation", "h30_d2y", sc, cfg)
    _, b = SE.run_group_trial_with_series("estimation", "h30_d2y", sc, cfg)
    for ra, rb in zip(a, b):
        assert np.array_equal(ra["vel_true"], rb["vel_true"])
        assert np.array_equal(ra["vel_est"], rb["vel_est"])
        assert ra["vel_meas"] is None or np.array_equal(ra["vel_meas"], rb["vel_meas"])


def test_plot_speed_diagnostics_saves_file(failed_trial_series, tmp_path):
    trial, series = failed_trial_series
    sc = SCENARIOS[1][1]
    path = tmp_path / "diag.png"
    D.plot_speed_diagnostics(trial, series, sc, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_failure_row_and_csv(failed_trial_series, tmp_path):
    trial, series = failed_trial_series
    m = D.lag_metrics(trial, series)
    row = D.failure_row(trial, m)
    sc = SCENARIOS[1][1]
    path = tmp_path / "fail.csv"
    D.write_failures_csv([row], str(path))
    text = path.read_text().splitlines()
    assert text[0].split(",") == list(row.keys())
    assert "h30_d2y" in text[1]


def test_scenario_by_name_unknown_raises():
    with pytest.raises(KeyError):
        D.scenario_by_name("missing", SCENARIOS)
    assert D.scenario_by_name("h20_d0", SCENARIOS) is SCENARIOS[0][1]