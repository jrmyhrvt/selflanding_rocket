"""
Robustness test for estimator variants under varying noise/dropout.

Tests alpha=0.25 (default, keep unchanged) vs alpha=0.7 under multiple
sensor configurations. All runs use the SAME scenarios and seeds as the
existing sensor experiment to keep comparisons fair.

No controller/physics changes. Purely virtual simulation.
"""
from dataclasses import dataclass, asdict, replace
import csv
import os
import numpy as np

from simulation.scenario import Scenario, run_scenario_with
from physics.controller import LandingController
from estimator.base import LowPassEstimator
from sensors.observed import SensorConfig, ObservedController
from evaluation import landing as L
from evaluation import sensor_experiment as SE

GROUPS = ("alpha_0p25", "alpha_0p7")

# Robustness configurations
ROBUST_CONFIGS = [
    ("base", SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, pos_bias_m=0.0, vel_bias_m_s=0.0, dropout_prob=0.1, seed=1234)),
    ("low_noise", SensorConfig(pos_std_m=0.5, vel_std_m_s=0.15, pos_bias_m=0.0, vel_bias_m_s=0.0, dropout_prob=0.1, seed=1234)),
    ("med_noise", SensorConfig(pos_std_m=1.5, vel_std_m_s=0.45, pos_bias_m=0.0, vel_bias_m_s=0.0, dropout_prob=0.1, seed=1234)),
    ("high_noise", SensorConfig(pos_std_m=2.0, vel_std_m_s=0.6, pos_bias_m=0.0, vel_bias_m_s=0.0, dropout_prob=0.1, seed=1234)),
    ("low_dropout", SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.05, seed=1234)),
    ("med_dropout", SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.15, seed=1234)),
    ("high_dropout", SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.25, seed=1234)),
    ("noisy_highdrop", SensorConfig(pos_std_m=1.5, vel_std_m_s=0.45, dropout_prob=0.2, seed=1234)),
]

SCENARIOS = SE.COMPARISON_SCENARIOS
SEEDS = SE.DEFAULT_SEEDS


@dataclass
class RobustTrial:
    group: str
    cfg_name: str
    scenario: str
    seed: int
    status: str
    success: bool
    v_end_m_s: float
    contact: bool
    target_dist_m: float
    est_pos_rms_m: float
    est_vel_rms_m_s: float
    missing_samples: int
    duration_s: float
    n_steps: int


def run_robust_trial(group: str, cfg_name: str, cfg: SensorConfig, name: str, scenario: Scenario, seed: int) -> RobustTrial:
    cfg_run = replace(cfg, seed=int(seed))
    inner = LandingController(descent_speed_m_s=scenario.descent_speed_m_s)
    alpha = 0.25 if group == "alpha_0p25" else 0.7
    est = LowPassEstimator(alpha_pos=alpha, alpha_vel=alpha)
    obs = ObservedController(inner, cfg_run, estimator=est)
    hist = run_scenario_with(scenario, obs)
    res = L.evaluate_landing(hist, scenario.target())
    last = hist[-1]
    epos, evel = obs.rms_errors()
    return RobustTrial(
        group=group,
        cfg_name=cfg_name,
        scenario=name,
        seed=int(seed),
        status=res["status"],
        success=(res["status"] == L.LANDED),
        v_end_m_s=float(np.linalg.norm(last.vel)),
        contact=bool(res["contact"]),
        target_dist_m=float(res["dist_xy"]),
        est_pos_rms_m=float(epos),
        est_vel_rms_m_s=float(evel),
        missing_samples=int(obs.samples_missing),
        duration_s=float(last.t),
        n_steps=len(hist),
    )


def run_robust_test(scenarios=None, configs=None, seeds=None, groups=None) -> list:
    scenarios = scenarios or SCENARIOS
    configs = configs or ROBUST_CONFIGS
    seeds = seeds or SEEDS
    groups = groups or GROUPS
    trials = []
    for cfg_name, cfg in configs:
        for g in groups:
            for name, sc in scenarios:
                for s in seeds:
                    trials.append(run_robust_trial(g, cfg_name, cfg, name, sc, s))
    return trials


def summarize_robust(trials) -> dict:
    out = {}
    for t in trials:
        key = (t.group, t.cfg_name)
        if key not in out:
            out[key] = {
                "trials": 0,
                "landed": 0,
                "too_fast": 0,
                "failed": 0,
                "no_data": 0,
                "v_end_mean": [],
                "v_end_max": [],
                "epos": [],
                "evel": [],
                "missing": [],
            }
        d = out[key]
        d["trials"] += 1
        if t.success: d["landed"] += 1
        if t.status == L.TOO_FAST: d["too_fast"] += 1
        if t.status == L.FAILED: d["failed"] += 1
        if t.status == L.NO_DATA: d["no_data"] += 1
        d["v_end_mean"].append(t.v_end_m_s)
        d["v_end_max"].append(t.v_end_m_s)
        d["epos"].append(t.est_pos_rms_m)
        d["evel"].append(t.est_vel_rms_m_s)
        d["missing"].append(t.missing_samples)
    for key in out:
        d = out[key]
        d["v_end_mean"] = float(np.mean(d["v_end_mean"])) if d["v_end_mean"] else float("nan")
        d["v_end_max"] = float(np.max(d["v_end_max"])) if d["v_end_max"] else float("nan")
        d["epos_mean"] = float(np.mean(d["epos"])) if d["epos"] else float("nan")
        d["evel_mean"] = float(np.mean(d["evel"])) if d["evel"] else float("nan")
        d["missing_mean"] = float(np.mean(d["missing"])) if d["missing"] else float("nan")
        d["success_rate"] = d["landed"] / d["trials"] if d["trials"] > 0 else 0.0
    return out


def write_robust_csv(trials, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(trials[0]).keys()) if trials else ["group"])
        w.writeheader()
        for t in trials:
            w.writerow(asdict(t))
    return path


def write_summary_robust(summary, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = sorted(summary.keys())
    fields = ["group", "cfg_name", "trials", "landed", "too_fast", "failed", "no_data", "success_rate",
              "v_end_mean", "v_end_max", "epos_mean", "evel_mean", "missing_mean"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for k in keys:
            group, cfg_name = k
            d = summary[k]
            row = {"group": group, "cfg_name": cfg_name}
            for f2 in fields[2:]:
                row[f2] = d.get(f2)
            w.writerow(row)
    return path


def plot_robust(summary, path: str = None, show: bool = False):
    import matplotlib
    if path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfgs = sorted(set(k[1] for k in summary.keys()))
    x = np.arange(len(cfgs))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # success rate
    sr25 = [summary.get(("alpha_0p25", c), {}).get("success_rate", 0.0) for c in cfgs]
    sr7 = [summary.get(("alpha_0p7", c), {}).get("success_rate", 0.0) for c in cfgs]
    w = 0.35
    ax1.bar(x - w / 2, sr25, w, label="alpha=0.25", color="tab:blue", alpha=0.7)
    ax1.bar(x + w / 2, sr7, w, label="alpha=0.7", color="tab:orange", alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(cfgs, rotation=45, ha="right")
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Success rate")
    ax1.set_title("Success rate by config"); ax1.legend(); ax1.grid(True, axis="y", alpha=0.3)

    # too_fast count
    tf25 = [summary.get(("alpha_0p25", c), {}).get("too_fast", 0.0) for c in cfgs]
    tf7 = [summary.get(("alpha_0p7", c), {}).get("too_fast", 0.0) for c in cfgs]
    ax2.bar(x - w / 2, tf25, w, label="alpha=0.25", color="tab:blue", alpha=0.7)
    ax2.bar(x + w / 2, tf7, w, label="alpha=0.7", color="tab:orange", alpha=0.7)
    ax2.set_xticks(x); ax2.set_xticklabels(cfgs, rotation=45, ha="right")
    ax2.set_ylabel("TOO_FAST count"); ax2.set_title("TOO_FAST by config"); ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)

    # est vel RMS
    ev25 = [summary.get(("alpha_0p25", c), {}).get("evel_mean", np.nan) for c in cfgs]
    ev7 = [summary.get(("alpha_0p7", c), {}).get("evel_mean", np.nan) for c in cfgs]
    ax3.bar(x - w / 2, ev25, w, label="alpha=0.25", color="tab:blue", alpha=0.7)
    ax3.bar(x + w / 2, ev7, w, label="alpha=0.7", color="tab:orange", alpha=0.7)
    ax3.set_xticks(x); ax3.set_xticklabels(cfgs, rotation=45, ha="right")
    ax3.set_ylabel("est vel RMS [m/s]"); ax3.set_title("Estimation error (vel)"); ax3.legend(); ax3.grid(True, axis="y", alpha=0.3)

    # missing mean
    miss25 = [summary.get(("alpha_0p25", c), {}).get("missing_mean", np.nan) for c in cfgs]
    miss7 = [summary.get(("alpha_0p7", c), {}).get("missing_mean", np.nan) for c in cfgs]
    ax4.bar(x - w / 2, miss25, w, label="alpha=0.25", color="tab:blue", alpha=0.7)
    ax4.bar(x + w / 2, miss7, w, label="alpha=0.7", color="tab:orange", alpha=0.7)
    ax4.set_xticks(x); ax4.set_xticklabels(cfgs, rotation=45, ha="right")
    ax4.set_ylabel("missing samples (mean)"); ax4.set_title("Dropouts"); ax4.legend(); ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Robustness: alpha=0.25 vs 0.7 under varying noise/dropout", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def main(outdir: str = "output/estimator_robustness", show: bool = False, configs=None, seeds=None):
    trials = run_robust_test(configs=configs, seeds=seeds)
    summary = summarize_robust(trials)
    write_robust_csv(trials, os.path.join(outdir, "robust_trials.csv"))
    write_summary_robust(summary, os.path.join(outdir, "robust_summary.csv"))
    plot_robust(summary, os.path.join(outdir, "robust_comparison.png"), show=show)
    return trials, summary
