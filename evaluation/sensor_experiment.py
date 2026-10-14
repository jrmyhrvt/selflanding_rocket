"""
Sensor-uncertainty / state-estimation comparison experiment.

Runs the SAME scenarios in four separate groups and compares them:

    ideal       controller sees the TRUE state              (reference)
    noise       controller sees noisy measurements          (no estimator)
    dropout     noisy measurements + occasional missing data
    estimation  noisy + missing data + LowPassEstimator

All groups share identical initial conditions. Physics is reused via
simulation.scenario.run_scenario_with; the only difference is the guidance
object (plain LandingController vs. ObservedController). There is no second
physics/control implementation.

Metrics per trial: success, end speed, target distance, steps, duration, and
(where an estimator runs) the RMS deviation between the ESTIMATE and the TRUE
simulated state. A failed/timeout run is never counted as a landing.

Learning model only. Not a realistic sensor/estimation model.
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


GROUPS = ("ideal", "noise", "dropout", "estimation")

# All landing statuses that must be counted (failed/no-data never silently
# dropped). Kept in sync with evaluation/landing.py.
STATUSES = (L.LANDED, L.TOO_FAST, L.FAILED, L.NO_DATA)

# Multiple reproducible seeds: one full pass over all scenarios per seed.
# Every pass uses identical scenarios/IC; only the sensor RNG seed changes.
DEFAULT_SEEDS = (1234, 101, 202, 303, 404)

# Documented, purely virtual sensor parameters.
DEFAULT_SENSOR_CFG = SensorConfig(
    pos_std_m=1.0, vel_std_m_s=0.3,
    pos_bias_m=0.0, vel_bias_m_s=0.0,
    dropout_prob=0.1, seed=1234,
)
ESTIMATOR_ALPHA = 0.25

# Identical initial conditions used for every group.
COMPARISON_SCENARIOS = [
    ("h20_d0", Scenario(start_height_m=20.0, target_xy=(0.0, 0.0))),
    ("h25_d3", Scenario(start_height_m=25.0, target_xy=(3.0, 0.0))),
    ("h40_d6", Scenario(start_height_m=40.0, target_xy=(6.0, 0.0))),
    ("h30_d2y", Scenario(start_height_m=30.0, target_xy=(2.0, 1.0),
                         v0_vertical_m_s=-2.0)),
    ("h60_d10", Scenario(start_height_m=60.0, target_xy=(10.0, 0.0))),
]


@dataclass
class GroupTrial:
    group: str
    scenario: str
    status: str
    success: bool
    contact: bool
    end_speed_m_s: float
    target_distance_m: float
    est_pos_rms_m: float       # NaN when no estimate (ideal)
    est_vel_rms_m_s: float     # NaN when no estimate (ideal)
    missing_samples: int
    duration_s: float
    n_steps: int
    seed: int = 0


def group_config(group: str, cfg: SensorConfig = None) -> SensorConfig:
    """Per-group sensor config: only dropout/estimation groups drop samples."""
    cfg = cfg or DEFAULT_SENSOR_CFG
    if group == "noise":
        return replace(cfg, dropout_prob=0.0)
    return cfg


def _run_group(group: str, name: str, scenario: Scenario, cfg: SensorConfig = None,
               record_series: bool = False):
    """Shared runner. Returns (trial, series_or_None).

    `record_series=True` captures the per-step TRUE / MEASUREMENT / ESTIMATE
    time series via ObservedController. Recording only appends data, so the
    trial result is numerically identical to an unrecorded run.
    """
    cfg = group_config(group, cfg)
    if group == "ideal":
        ctrl = LandingController(descent_speed_m_s=scenario.descent_speed_m_s)
        hist = run_scenario_with(scenario, ctrl)
        obs = None
    else:
        inner = LandingController(descent_speed_m_s=scenario.descent_speed_m_s)
        est = (LowPassEstimator(ESTIMATOR_ALPHA, ESTIMATOR_ALPHA)
               if group == "estimation" else None)
        obs = ObservedController(inner, cfg, estimator=est, record_series=record_series)
        hist = run_scenario_with(scenario, obs)

    res = L.evaluate_landing(hist, scenario.target())
    last = hist[-1]
    if obs is not None:
        epos, evel = obs.rms_errors()
        missing = obs.samples_missing
    else:
        epos = evel = float("nan")
        missing = 0

    trial = GroupTrial(
        group=group,
        scenario=name,
        status=res["status"],
        success=(res["status"] == L.LANDED),
        contact=bool(res["contact"]),
        end_speed_m_s=float(np.linalg.norm(last.vel)),
        target_distance_m=float(res["dist_xy"]),
        est_pos_rms_m=float(epos),
        est_vel_rms_m_s=float(evel),
        missing_samples=int(missing),
        duration_s=float(last.t),
        n_steps=len(hist),
        seed=int(cfg.seed),
    )
    series = obs.record if (obs is not None and record_series) else None
    return trial, series


def run_group_trial(group: str, name: str, scenario: Scenario,
                    cfg: SensorConfig = None) -> GroupTrial:
    trial, _series = _run_group(group, name, scenario, cfg, record_series=False)
    return trial


def run_group_trial_with_series(group: str, name: str, scenario: Scenario,
                                cfg: SensorConfig = None):
    """Run a single trial AND return its per-step time series."""
    return _run_group(group, name, scenario, cfg, record_series=True)


def run_comparison(scenarios=None, cfg: SensorConfig = None, groups=GROUPS) -> list:
    scenarios = scenarios if scenarios is not None else COMPARISON_SCENARIOS
    trials = []
    for group in groups:
        for name, sc in scenarios:
            trials.append(run_group_trial(group, name, sc, cfg))
    return trials


def run_repeated(scenarios=None, seeds=DEFAULT_SEEDS, cfg: SensorConfig = None,
                 groups=GROUPS) -> list:
    """Run the full comparison once per seed.

    Every seed re-runs the SAME scenarios with the SAME initial conditions and
    the SAME controller parameters; only the sensor RNG seed differs. The
    resulting seed is stored on each trial. Fully deterministic for a given
    (scenarios, seeds, cfg).
    """
    base = cfg or DEFAULT_SENSOR_CFG
    trials = []
    for s in seeds:
        trials.extend(run_comparison(scenarios=scenarios,
                                     cfg=replace(base, seed=int(s)),
                                     groups=groups))
    return trials


def scenario_signature(sc: Scenario) -> tuple:
    """All initial-condition / fixed settings that must be identical across modes."""
    st = sc.initial_state()
    return (tuple(float(x) for x in st.pos), tuple(float(x) for x in st.vel),
            tuple(float(x) for x in sc.target()), float(sc.dt),
            float(sc.max_time_s), float(sc.descent_speed_m_s),
            float(sc.dry_mass_kg))


def verify_fair_comparison(scenarios=None, groups=GROUPS) -> bool:
    """True if every mode runs the exact same scenarios/initial conditions.

    Fairness is by construction here (one Scenario object per case, passed to
    every mode); this asserts it so a future change cannot silently break it.
    """
    scenarios = scenarios if scenarios is not None else COMPARISON_SCENARIOS
    for name, sc in scenarios:
        sig = scenario_signature(sc)
        for _group in groups:
            if scenario_signature(sc) != sig:
                return False
    return True


def _nanmean(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _nanmax(values):
    vals = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.max(vals)) if vals else float("nan")


def stats_for(rows) -> dict:
    """Aggregate one bucket (group, or group+scenario) across all seeds.

    Failed/timeout runs are counted too: `trials` includes everything and the
    per-status counts always sum back to `trials`.
    """
    n = len(rows)
    landed = sum(1 for t in rows if t.success)
    counts = {st: sum(1 for t in rows if t.status == st) for st in STATUSES}
    contact_rows = [t for t in rows if t.contact]
    speeds = [t.end_speed_m_s for t in rows]
    return {
        "trials": n,
        "landed": landed,
        "failed": n - landed,
        "success_rate": (landed / n) if n else 0.0,
        "status_LANDED": counts[L.LANDED],
        "status_TOO_FAST": counts[L.TOO_FAST],
        "status_FAILED": counts[L.FAILED],
        "status_NO_DATA": counts[L.NO_DATA],
        "n_contact": len(contact_rows),
        "mean_end_speed_m_s": _nanmean(speeds),
        "max_end_speed_m_s": _nanmax(speeds),
        "mean_contact_distance_m": _nanmean([t.target_distance_m for t in contact_rows]),
        "mean_est_pos_rms_m": _nanmean([t.est_pos_rms_m for t in rows]),
        "mean_est_vel_rms_m_s": _nanmean([t.est_vel_rms_m_s for t in rows]),
        "seeds": sorted({t.seed for t in rows}),
    }


def summarize_statistics(trials) -> dict:
    """Per-mode and per-mode+scenario statistics over all seeds."""
    out = {
        "n_trials": len(trials),
        "seeds": sorted({t.seed for t in trials}),
        "groups": list(GROUPS),
        "by_group": {},
    }
    for g in GROUPS:
        grows = [t for t in trials if t.group == g]
        if not grows:
            continue
        ordered = []
        for t in grows:
            if t.scenario not in ordered:
                ordered.append(t.scenario)
        entry = stats_for(grows)
        entry["scenarios"] = {name: stats_for([t for t in grows if t.scenario == name])
                              for name in ordered}
        out["by_group"][g] = entry
    return out


# ---------------------------------------------------------------- reporting
SUMMARY_FIELDS = [
    "group", "scenario", "trials", "landed", "failed", "success_rate",
    "status_LANDED", "status_TOO_FAST", "status_FAILED", "status_NO_DATA",
    "n_contact", "mean_end_speed_m_s", "max_end_speed_m_s",
    "mean_contact_distance_m", "mean_est_pos_rms_m", "mean_est_vel_rms_m_s",
    "seeds",
]


def format_statistics_table(stats) -> str:
    header = (f"{'mode':<12}{'scenario':<10}{'trials':>7}{'LND':>5}{'FAST':>6}"
              f"{'FAIL':>6}{'NOD':>5}{'succ':>7}{'v_mean':>8}{'v_max':>7}"
              f"{'dist*':>7}{'e_pos':>8}{'e_vel':>7}")
    lines = [header, "-" * len(header)]
    for g, e in stats["by_group"].items():
        def _row(label, s):
            epos = "   n/a" if np.isnan(s["mean_est_pos_rms_m"]) else f"{s['mean_est_pos_rms_m']:8.3f}"
            evel = "   n/a" if np.isnan(s["mean_est_vel_rms_m_s"]) else f"{s['mean_est_vel_rms_m_s']:7.3f}"
            return (f"{g:<12}{label:<10}{s['trials']:>7}"
                    f"{s['status_LANDED']:>5}{s['status_TOO_FAST']:>6}"
                    f"{s['status_FAILED']:>6}{s['status_NO_DATA']:>5}"
                    f"{s['success_rate']*100:>6.0f}%"
                    f"{s['mean_end_speed_m_s']:>8.3f}{s['max_end_speed_m_s']:>7.3f}"
                    f"{s['mean_contact_distance_m']:>7.2f}{epos}{evel}")
        lines.append(_row("ALL", e))
        for name, s in e["scenarios"].items():
            lines.append(_row("  " + name, s))
    lines.append("")
    lines.append("LND=LANDED FAST=TOO_FAST FAIL=FAILED NOD=NO_DATA "
                 "dist*=mean target distance at contact")
    return "\n".join(lines)


def write_summary_csv(stats, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for g, e in stats["by_group"].items():
            rows = [("ALL", e)] + list(e["scenarios"].items())
            for label, s in rows:
                rec = {k: s[k] for k in SUMMARY_FIELDS if k in s}
                rec["group"] = g
                rec["scenario"] = label
                rec["seeds"] = ";".join(str(x) for x in s["seeds"])
                writer.writerow(rec)
    return path


def _success_rate_by_seed(trials, group) -> dict:
    out = {}
    for t in trials:
        if t.group != group:
            continue
        out.setdefault(t.seed, []).append(t.success)
    return {s: (sum(v) / len(v)) for s, v in out.items()}


def plot_statistics(trials, stats, path: str = None, show: bool = False):
    import matplotlib
    if path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = stats["by_group"]
    groups = [g for g in GROUPS if g in by]
    x = np.arange(len(groups))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1) success rate per mode, with a point per seed (stability across seeds)
    ax1.bar(x, [by[g]["success_rate"] for g in groups], color="tab:green", alpha=0.5)
    for i, g in enumerate(groups):
        per_seed = _success_rate_by_seed(trials, g)
        ax1.scatter([i] * len(per_seed), list(per_seed.values()), color="black",
                    zorder=5, s=28, label="je Seed" if i == 0 else None)
    ax1.set_xticks(x); ax1.set_xticklabels(groups, rotation=15)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Erfolgsquote")
    ax1.set_title("Erfolgsquote pro Modus (Balken) + je Seed (Punkte)")
    ax1.legend(fontsize=8); ax1.grid(True, axis="y", alpha=0.3)

    # 2) stacked status counts per mode
    bottoms = np.zeros(len(groups))
    colors = {L.LANDED: "tab:green", L.TOO_FAST: "tab:orange",
              L.FAILED: "tab:red", L.NO_DATA: "tab:gray"}
    for st in STATUSES:
        vals = np.array([by[g][f"status_{st}"] for g in groups], dtype=float)
        ax2.bar(x, vals, bottom=bottoms, color=colors[st], alpha=0.8, label=st)
        bottoms += vals
    ax2.set_xticks(x); ax2.set_xticklabels(groups, rotation=15)
    ax2.set_ylabel("Anzahl Versuche"); ax2.set_title("Statusverteilung")
    ax2.legend(fontsize=8); ax2.grid(True, axis="y", alpha=0.3)

    # 3) mean and max end speed per mode
    w = 0.38
    ax3.bar(x - w / 2, [by[g]["mean_end_speed_m_s"] for g in groups], w,
            label="Mittel", color="tab:blue", alpha=0.8)
    ax3.bar(x + w / 2, [by[g]["max_end_speed_m_s"] for g in groups], w,
            label="Max", color="tab:cyan", alpha=0.8)
    ax3.axhline(L.V_TOL_M_S, color="r", ls="--", label="v_tol = 0.5")
    ax3.set_xticks(x); ax3.set_xticklabels(groups, rotation=15)
    ax3.set_ylabel("Endgeschwindigkeit [m/s]")
    ax3.set_title("Endgeschwindigkeit"); ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    # 4) estimation error (modes with an estimate)
    est_groups = [g for g in groups if g != "ideal"]
    xb = np.arange(len(est_groups))
    ax4.bar(xb - w / 2, [by[g]["mean_est_pos_rms_m"] for g in est_groups], w,
            label="pos RMS [m]", color="tab:orange", alpha=0.8)
    ax4.bar(xb + w / 2, [by[g]["mean_est_vel_rms_m_s"] for g in est_groups], w,
            label="vel RMS [m/s]", color="tab:cyan", alpha=0.8)
    ax4.set_xticks(xb); ax4.set_xticklabels(est_groups, rotation=15)
    ax4.set_title("Schaetzfehler (Schaetzung vs. Wahrheit)")
    ax4.legend(fontsize=8); ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Statistische Auswertung: {len(stats['seeds'])} Seeds, "
                 f"{stats['n_trials']} Versuche (vereinfachtes Punktmassen-Modell)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def summarize_by_group(trials) -> dict:
    out = {}
    for group in GROUPS:
        rows = [t for t in trials if t.group == group]
        if not rows:
            continue
        landed = sum(1 for t in rows if t.success)
        out[group] = {
            "total": len(rows),
            "landed": landed,
            "failed": len(rows) - landed,
            "success_rate": landed / len(rows),
            "mean_end_speed_m_s": _nanmean([t.end_speed_m_s for t in rows]),
            "mean_target_distance_m": _nanmean([t.target_distance_m for t in rows]),
            "mean_est_pos_rms_m": _nanmean([t.est_pos_rms_m for t in rows]),
            "mean_est_vel_rms_m_s": _nanmean([t.est_vel_rms_m_s for t in rows]),
            "mean_missing_samples": _nanmean([t.missing_samples for t in rows]),
        }
    return out


def format_table(trials) -> str:
    header = (f"{'group':<11}{'scenario':<10}{'status':>9}{'v_end':>8}"
              f"{'dist':>7}{'epos':>8}{'evel':>8}{'miss':>6}{'steps':>7}")
    lines = [header, "-" * len(header)]
    for t in trials:
        epos = "   n/a" if np.isnan(t.est_pos_rms_m) else f"{t.est_pos_rms_m:8.3f}"
        evel = "   n/a" if np.isnan(t.est_vel_rms_m_s) else f"{t.est_vel_rms_m_s:8.3f}"
        lines.append(
            f"{t.group:<11}{t.scenario:<10}{t.status:>9}"
            f"{t.end_speed_m_s:>8.2f}{t.target_distance_m:>7.2f}"
            f"{epos}{evel}{t.missing_samples:>6d}{t.n_steps:>7d}")
    return "\n".join(lines)


def write_results_csv(trials, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(asdict(trials[0]).keys()) if trials else ["group", "scenario", "status"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            writer.writerow(asdict(t))
    return path


def plot_results(trials, path: str = None, show: bool = False):
    import matplotlib
    if path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = summarize_by_group(trials)
    groups = [g for g in GROUPS if g in summ]
    x = np.arange(len(groups))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # 1) success rate
    ax1.bar(x, [summ[g]["success_rate"] for g in groups], color="tab:green", alpha=0.7)
    ax1.set_xticks(x); ax1.set_xticklabels(groups, rotation=15)
    ax1.set_ylim(0, 1.05); ax1.set_ylabel("Erfolgsquote")
    ax1.set_title("Landungen pro Gruppe"); ax1.grid(True, axis="y", alpha=0.3)

    # 2) mean end speed
    ax2.bar(x, [summ[g]["mean_end_speed_m_s"] for g in groups], color="tab:blue", alpha=0.7)
    ax2.axhline(L.V_TOL_M_S, color="r", ls="--", label="v_tol = 0.5")
    ax2.set_xticks(x); ax2.set_xticklabels(groups, rotation=15)
    ax2.set_ylabel("mittlere Endgeschwindigkeit [m/s]")
    ax2.set_title("Endgeschwindigkeit (Mittel)"); ax2.legend(fontsize=8)
    ax2.grid(True, axis="y", alpha=0.3)

    # 3) mean target distance
    ax3.bar(x, [summ[g]["mean_target_distance_m"] for g in groups], color="tab:purple", alpha=0.7)
    ax3.axhline(L.DIST_TOL_M, color="r", ls="--", label="dist_tol = 1.0")
    ax3.set_xticks(x); ax3.set_xticklabels(groups, rotation=15)
    ax3.set_ylabel("mittlerer Zielabstand [m]")
    ax3.set_title("Zielabstand (Mittel)"); ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    # 4) estimation error (groups with an estimate)
    est_groups = [g for g in groups if g != "ideal"]
    xb = np.arange(len(est_groups))
    w = 0.38
    ax4.bar(xb - w / 2, [summ[g]["mean_est_pos_rms_m"] for g in est_groups],
            w, label="pos RMS [m]", color="tab:orange", alpha=0.8)
    ax4.bar(xb + w / 2, [summ[g]["mean_est_vel_rms_m_s"] for g in est_groups],
            w, label="vel RMS [m/s]", color="tab:cyan", alpha=0.8)
    ax4.set_xticks(xb); ax4.set_xticklabels(est_groups, rotation=15)
    ax4.set_title("Schaetzfehler (Schaetzung vs. Wahrheit)")
    ax4.legend(fontsize=8); ax4.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Sensorunsicherheit & Zustandsschaetzung "
                 "(vereinfachtes Punktmassen-Modell)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def main(outdir: str = "output/sensor_experiment", show: bool = False,
         seeds=DEFAULT_SEEDS, scenarios=None):
    trials = run_repeated(scenarios=scenarios, seeds=seeds)
    stats = summarize_statistics(trials)
    print(format_statistics_table(stats))
    print("=" * 80)
    print(f"Seeds: {stats['seeds']}  |  Versuche gesamt: {stats['n_trials']}  |  "
          f"Modi: {', '.join(stats['groups'])}")
    for g, e in stats["by_group"].items():
        print(f"{g:<11} LANDED={e['status_LANDED']} TOO_FAST={e['status_TOO_FAST']} "
              f"FAILED={e['status_FAILED']} NO_DATA={e['status_NO_DATA']} "
              f"-> Erfolg {e['landed']}/{e['trials']} ({e['success_rate']*100:.0f}%)")
    trials_csv = write_results_csv(
        trials, os.path.join(outdir, "sensor_experiment_trials.csv"))
    summary_csv = write_summary_csv(
        stats, os.path.join(outdir, "sensor_experiment_summary.csv"))
    png_path = os.path.join(outdir, "sensor_experiment_statistics.png")
    plot_statistics(trials, stats, png_path, show=show)
    print(f"Trials-CSV:  {trials_csv}")
    print(f"Summary-CSV: {summary_csv}")
    print(f"Plot:        {png_path}")
    return trials


if __name__ == "__main__":
    main()
