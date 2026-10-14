from dataclasses import asdict
import numpy as np
import pytest

from simulation.scenario import Scenario, run_scenario, run_scenario_with
from physics.controller import LandingController
from evaluation import landing as L
from evaluation import sensor_experiment as SE


SMALL = [
    ("h20_d0", Scenario(start_height_m=20.0, target_xy=(0.0, 0.0))),
    ("h25_d3", Scenario(start_height_m=25.0, target_xy=(3.0, 0.0))),
]
SEEDS = (1234, 7)


@pytest.fixture(scope="module")
def trials():
    return SE.run_repeated(scenarios=SMALL, seeds=SEEDS)


@pytest.fixture(scope="module")
def stats(trials):
    return SE.summarize_statistics(trials)


def _same(a, b):
    da, db = asdict(a), asdict(b)
    assert da.keys() == db.keys()
    for k in da:
        va, vb = da[k], db[k]
        if isinstance(va, float) and isinstance(vb, float):
            assert (np.isnan(va) and np.isnan(vb)) or va == vb
        else:
            assert va == vb


def test_multiseed_count_and_seed_tags(trials):
    expected = len(SMALL) * len(SE.GROUPS) * len(SEEDS)
    assert len(trials) == expected
    assert {t.seed for t in trials} == set(SEEDS)


def test_multiseed_reproducible():
    a = SE.run_repeated(scenarios=SMALL, seeds=SEEDS)
    b = SE.run_repeated(scenarios=SMALL, seeds=SEEDS)
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        _same(ta, tb)


def test_different_seeds_change_noisy_runs_but_not_ideal():
    one = SE.run_repeated(scenarios=SMALL, seeds=(1,))
    two = SE.run_repeated(scenarios=SMALL, seeds=(2,))

    def speeds(rows, group, name):
        return [t.end_speed_m_s for t in rows if t.group == group and t.scenario == name]

    for name, _sc in SMALL:
        # ideal has no randomness -> identical across seeds
        assert speeds(one, "ideal", name) == speeds(two, "ideal", name)
    # noisy group must actually vary when only the seed changes
    changed = any(speeds(one, "noise", n) != speeds(two, "noise", n)
                  for n, _ in SMALL)
    assert changed


def test_fair_comparison_identical_ics(trials):
    assert SE.verify_fair_comparison(SMALL) is True
    for group in SE.GROUPS:
        for name, _sc in SMALL:
            rows = [t for t in trials if t.group == group and t.scenario == name]
            assert len(rows) == len(SEEDS)


def test_noise_has_no_dropout_dropout_has_dropout(trials):
    noise_missing = [t.missing_samples for t in trials if t.group == "noise"]
    assert all(m == 0 for m in noise_missing)
    for group in ("dropout", "estimation"):
        assert sum(t.missing_samples for t in trials if t.group == group) > 0


def test_status_counts_sum_to_trials(stats):
    for g, e in stats["by_group"].items():
        total = (e["status_LANDED"] + e["status_TOO_FAST"]
                 + e["status_FAILED"] + e["status_NO_DATA"])
        assert total == e["trials"]
        assert e["landed"] == e["status_LANDED"]
        assert e["failed"] == e["trials"] - e["landed"]
        assert np.isclose(e["success_rate"], e["landed"] / e["trials"])
        scen_total = sum(s["trials"] for s in e["scenarios"].values())
        assert scen_total == e["trials"]


def test_failed_runs_are_not_dropped():
    timed_out = [("timeout", Scenario(start_height_m=50.0, target_xy=(0.0, 0.0),
                                      max_time_s=1.0))]
    trials = SE.run_repeated(scenarios=timed_out, seeds=(1,), groups=("ideal",))
    st = SE.summarize_statistics(trials)
    e = st["by_group"]["ideal"]
    assert e["trials"] == 1
    assert e["status_FAILED"] == 1
    assert e["status_LANDED"] == 0
    assert e["landed"] == 0
    assert e["success_rate"] == 0.0


def test_mean_not_above_max_end_speed(stats):
    for g, e in stats["by_group"].items():
        if np.isfinite(e["mean_end_speed_m_s"]):
            assert e["mean_end_speed_m_s"] <= e["max_end_speed_m_s"] + 1e-12


def test_seeds_recorded_in_statistics(stats):
    assert stats["seeds"] == sorted(SEEDS)
    for e in stats["by_group"].values():
        assert e["seeds"] == sorted(SEEDS)


def test_estimation_error_only_where_measurable(stats):
    assert np.isnan(stats["by_group"]["ideal"]["mean_est_pos_rms_m"])
    assert np.isnan(stats["by_group"]["ideal"]["mean_est_vel_rms_m_s"])
    for g in ("noise", "dropout", "estimation"):
        assert np.isfinite(stats["by_group"][g]["mean_est_pos_rms_m"])
        assert np.isfinite(stats["by_group"][g]["mean_est_vel_rms_m_s"])


def test_mean_contact_distance_only_over_contact(stats):
    for e in stats["by_group"].values():
        if e["n_contact"] == 0:
            assert np.isnan(e["mean_contact_distance_m"])
        else:
            assert e["mean_contact_distance_m"] >= 0.0


def test_ideal_group_full_history_matches_reference():
    sc = SMALL[0][1]
    ref = run_scenario(sc)
    ctrl = LandingController(descent_speed_m_s=sc.descent_speed_m_s)
    hist = run_scenario_with(sc, ctrl)
    assert len(ref) == len(hist)
    for a, b in zip(ref, hist):
        assert np.allclose(a.pos, b.pos)
        assert np.allclose(a.vel, b.vel)


def test_summary_csv(tmp_path, stats):
    path = tmp_path / "sub" / "summary.csv"
    SE.write_summary_csv(stats, str(path))
    lines = path.read_text().splitlines()
    assert lines[0].split(",") == SE.SUMMARY_FIELDS
    expected_rows = sum(1 + len(e["scenarios"]) for e in stats["by_group"].values())
    assert len(lines) == expected_rows + 1
    assert any(row.startswith("ideal,ALL,") for row in lines[1:])
    assert any(row.startswith("noise,") for row in lines[1:])


def test_plot_statistics_saves_file(tmp_path, trials, stats):
    path = tmp_path / "stats.png"
    SE.plot_statistics(trials, stats, str(path))
    assert path.exists() and path.stat().st_size > 0


def test_format_statistics_table_contains_modes(stats):
    table = SE.format_statistics_table(stats)
    for g in SE.GROUPS:
        assert g in table
    assert "ALL" in table
