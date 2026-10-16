import numpy as np
from dataclasses import asdict
from evaluation import estimator_robustness as ER
from evaluation import sensor_experiment as SES

SMALL_CFGS = [
    ("base", ER.SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.1, seed=1234)),
    ("low_noise", ER.SensorConfig(pos_std_m=0.5, vel_std_m_s=0.15, dropout_prob=0.1, seed=1234)),
]
SMALL_SCEN = [SES.COMPARISON_SCENARIOS[0], SES.COMPARISON_SCENARIOS[1]]
SEEDS = (1234, 7)


def test_fair_same_scenarios_seeds():
    trials = ER.run_robust_test(scenarios=SMALL_SCEN, configs=SMALL_CFGS, seeds=SEEDS)
    expected = len(ER.GROUPS) * len(SMALL_CFGS) * len(SMALL_SCEN) * len(SEEDS)
    assert len(trials) == expected


def test_reproducible():
    a = ER.run_robust_test(scenarios=SMALL_SCEN, configs=SMALL_CFGS[:1], seeds=(1,))
    b = ER.run_robust_test(scenarios=SMALL_SCEN, configs=SMALL_CFGS[:1], seeds=(1,))
    assert [asdict(x) for x in a] == [asdict(x) for x in b]


def test_summary_counts_sum():
    trials = ER.run_robust_test(scenarios=SMALL_SCEN, configs=SMALL_CFGS[:1], seeds=(1,))
    s = ER.summarize_robust(trials)
    for (g, c), d in s.items():
        assert d["landed"] + d["too_fast"] + d["failed"] + d["no_data"] == d["trials"]


def test_higher_alpha_tracks_noise_more():
    trials = ER.run_robust_test(scenarios=[SMALL_SCEN[0]], configs=[("high_noise", ER.SensorConfig(pos_std_m=2.0, vel_std_m_s=0.6, dropout_prob=0.1, seed=1234))], seeds=(1234,))
    s = ER.summarize_robust(trials)
    a25 = s[("alpha_0p25", "high_noise")]["evel_mean"]
    a7 = s[("alpha_0p7", "high_noise")]["evel_mean"]
    assert a7 >= a25


def test_alpha7_missing_increases_with_dropout():
    trials = ER.run_robust_test(scenarios=[SMALL_SCEN[0]], configs=[("high_dropout", ER.SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.25, seed=1234))], seeds=(1234,))
    s = ER.summarize_robust(trials)
    a7m = s[("alpha_0p7", "high_dropout")]["missing_mean"]
    assert a7m > 0
