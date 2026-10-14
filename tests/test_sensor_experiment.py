from dataclasses import asdict
import numpy as np
import pytest

from simulation.scenario import Scenario, run_scenario, run_scenario_with
from physics.controller import LandingController
from sensors.observed import SensorConfig, ObservedController
from evaluation import landing as L
from evaluation import sensor_experiment as SE


SMALL = [
    ("h20_d0", Scenario(start_height_m=20.0, target_xy=(0.0, 0.0))),
    ("h25_d3", Scenario(start_height_m=25.0, target_xy=(3.0, 0.0))),
]


class _ConstController:
    """Ignores its input and always returns the same thrust command."""
    def __init__(self, cmd):
        self.cmd = np.asarray(cmd, dtype=float)

    def compute(self, state, target_pos, mass_kg):
        return self.cmd.copy()


def test_run_comparison_covers_all_groups():
    trials = SE.run_comparison(scenarios=SMALL)
    groups = {t.group for t in trials}
    assert groups == set(SE.GROUPS)
    assert len(trials) == len(SE.GROUPS) * len(SMALL)


def test_noise_group_has_no_dropout_dropout_group_does():
    trials = SE.run_comparison(scenarios=SMALL)
    for t in trials:
        if t.group == "noise":
            assert t.missing_samples == 0
        if t.group in ("dropout", "estimation"):
            assert t.missing_samples > 0


def test_ideal_group_has_no_estimate_metrics():
    trials = SE.run_comparison(scenarios=SMALL, groups=("ideal",))
    for t in trials:
        assert np.isnan(t.est_pos_rms_m)
        assert np.isnan(t.est_vel_rms_m_s)
        assert t.missing_samples == 0


def test_ideal_group_matches_reference_run():
    sc = Scenario(start_height_m=20.0, target_xy=(0.0, 0.0))
    ref = L.evaluate_landing(run_scenario(sc), sc.target())["status"]
    trial = SE.run_group_trial("ideal", "h20_d0", sc)
    assert trial.status == ref


def _same(a, b):
    da, db = asdict(a), asdict(b)
    assert da.keys() == db.keys()
    for k in da:
        va, vb = da[k], db[k]
        if isinstance(va, float) and isinstance(vb, float):
            assert (np.isnan(va) and np.isnan(vb)) or va == vb
        else:
            assert va == vb


def test_reproducible_with_seed():
    a = SE.run_comparison(scenarios=SMALL)
    b = SE.run_comparison(scenarios=SMALL)
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        _same(ta, tb)


def test_summary_structure_and_rates():
    trials = SE.run_comparison(scenarios=SMALL)
    summ = SE.summarize_by_group(trials)
    assert set(summ) == set(SE.GROUPS)
    for g, s in summ.items():
        assert s["total"] == len(SMALL)
        assert 0.0 <= s["success_rate"] <= 1.0
        assert s["landed"] + s["failed"] == s["total"]
    assert np.isnan(summ["ideal"]["mean_est_pos_rms_m"])
    assert np.isfinite(summ["noise"]["mean_est_pos_rms_m"])


def test_filter_reduces_estimation_error():
    trials = SE.run_comparison(scenarios=SMALL)
    summ = SE.summarize_by_group(trials)
    assert summ["estimation"]["mean_est_pos_rms_m"] < summ["noise"]["mean_est_pos_rms_m"]
    assert summ["estimation"]["mean_est_vel_rms_m_s"] < summ["noise"]["mean_est_vel_rms_m_s"]


def test_failed_run_is_not_success():
    sc = Scenario(start_height_m=50.0, target_xy=(0.0, 0.0), max_time_s=1.0)
    t = SE.run_group_trial("noise", "timeout", sc)
    assert t.success is False
    assert t.status != L.LANDED


def test_sensors_do_not_change_physics_when_controller_ignores_input():
    sc = Scenario(start_height_m=20.0, target_xy=(0.0, 0.0))
    cmd = [0.0, 0.0, 9.81]  # hover-ish, ignores measurements entirely
    cfg_a = SensorConfig(pos_std_m=0.0, vel_std_m_s=0.0, dropout_prob=0.0, seed=1)
    cfg_b = SensorConfig(pos_std_m=5.0, vel_std_m_s=2.0, dropout_prob=0.5, seed=2)
    h_a = run_scenario_with(sc, ObservedController(_ConstController(cmd), cfg_a))
    h_b = run_scenario_with(sc, ObservedController(_ConstController(cmd), cfg_b))
    assert len(h_a) == len(h_b)
    for sa, sb in zip(h_a, h_b):
        assert np.allclose(sa.pos, sb.pos)
        assert np.allclose(sa.vel, sb.vel)


def test_write_results_csv(tmp_path):
    trials = SE.run_comparison(scenarios=SMALL)
    path = tmp_path / "sub" / "out.csv"
    SE.write_results_csv(trials, str(path))
    text = path.read_text().splitlines()
    assert text[0].split(",") == list(asdict(trials[0]).keys())
    assert len(text) == len(trials) + 1


def test_plot_results_saves_file(tmp_path):
    trials = SE.run_comparison(scenarios=SMALL)
    path = tmp_path / "plot.png"
    SE.plot_results(trials, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_format_table_contains_groups():
    trials = SE.run_comparison(scenarios=SMALL, groups=("ideal",))
    table = SE.format_table(trials)
    assert "ideal" in table and "LANDED" in table
