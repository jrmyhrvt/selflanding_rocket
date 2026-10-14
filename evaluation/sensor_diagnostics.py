"""
Diagnosis of the estimation-mode failures (TOO_FAST) in the sensor experiment.

Goal (no optimisation, no controller change): show WHEN and by HOW MUCH the
estimated velocity deviates from the true simulated one.

Everything here uses REAL recorded simulation data:
  - the trial is run through simulation.scenario.run_scenario_with with the
    SAME scenario and seed as in the experiment,
  - the per-step time series comes from ObservedController.record (TRUE state,
    raw MEASUREMENT, ESTIMATE), which is captured during that very run.
No synthetic example curves are plotted.

Findings are reported per failed trial and a 3-panel figure (height, vertical
velocity, speed magnitude) marks ground contact and labels TOO_FAST.

Learning model only. Not a realistic sensor/estimation model. No hardware.
"""
from dataclasses import replace
import csv
import os
import numpy as np

from simulation.scenario import Scenario
from sensors.observed import SensorConfig
from evaluation import landing as L
from evaluation import sensor_experiment as SE


def find_failures(trials, statuses=(L.TOO_FAST, L.FAILED), groups=("estimation",)):
    """Select failed trials of the given modes/statuses (keeps all of them)."""
    return [t for t in trials
            if t.group in groups and t.status in statuses]


def scenario_by_name(name: str, scenarios=None) -> Scenario:
    scenarios = scenarios if scenarios is not None else SE.COMPARISON_SCENARIOS
    for n, sc in scenarios:
        if n == name:
            return sc
    raise KeyError(f"unknown scenario name '{name}'")


def lag_metrics(trial, series) -> dict:
    """Quantify when / how much the estimate deviates from the true state.

    All values come from the recorded series of the real run.
    """
    t = np.array([r["t"] for r in series])
    vz_true = np.array([r["vel_true"][2] for r in series])
    vz_est = np.array([r["vel_est"][2] for r in series])
    dev = vz_est - vz_true
    t_last = t[-1]
    last_2s = t >= (t_last - 2.0)
    first_dev = None
    for i, d in enumerate(dev):
        if abs(d) > 0.05:
            first_dev = float(t[i])
            break
    max_idx = int(np.argmax(np.abs(dev)))
    return {
        "t_contact_s": float(t_last),
        "vz_true_end_m_s": float(vz_true[-1]),
        "vz_est_end_m_s": float(vz_est[-1]),
        "vz_end_delta_m_s": float(vz_est[-1] - vz_true[-1]),
        "max_abs_dev_m_s": float(np.max(np.abs(dev))),
        "t_max_dev_s": float(t[max_idx]),
        "mean_abs_dev_run_m_s": float(np.mean(np.abs(dev))),
        "mean_abs_dev_last_2s_m_s": float(np.mean(np.abs(dev[last_2s]))),
        "first_dev_gt_0p05_s": first_dev,
    }


def failure_row(trial, metrics: dict) -> dict:
    row = {
        "group": trial.group,
        "scenario": trial.scenario,
        "seed": trial.seed,
        "status": trial.status,
        "v_end_m_s": trial.end_speed_m_s,
        "contact": trial.contact,
        "target_distance_m": trial.target_distance_m,
        "est_pos_rms_m": trial.est_pos_rms_m,
        "est_vel_rms_m_s": trial.est_vel_rms_m_s,
        "n_steps": trial.n_steps,
    }
    row.update(metrics)
    return row


def write_failures_csv(rows, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path


def plot_speed_diagnostics(trial, series, sc: Scenario, path: str = None,
                           show: bool = False):
    """3-panel diagnosis figure from the REAL recorded simulation run.

    Top    : height h(t)                       (true trajectory)
    Middle : vertical velocity vz: TRUE (line), MEASURED (dots, gaps =
             dropped samples), ESTIMATE (line)
    Bottom : speed |v| with the 0.5 m/s soft-landing limit

    Ground contact is marked with a vertical line on all panels and the status
    is labeled. No synthetic curves.
    """
    import matplotlib
    if path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([r["t"] for r in series])
    h = np.array([r["pos_true"][2] for r in series])
    vz_true = np.array([r["vel_true"][2] for r in series])
    vz_est = np.array([r["vel_est"][2] for r in series])
    vm_true = np.array([r["vel_meas"][2] for r in series
                        if r["vel_meas"] is not None])
    tm_valid = np.array([r["t"] for r in series if r["vel_meas"] is not None])
    sp_true = np.array([np.linalg.norm(r["vel_true"]) for r in series])
    sp_est = np.array([np.linalg.norm(r["vel_est"]) for r in series])
    sp_meas = np.array([np.linalg.norm(r["vel_meas"]) for r in series
                        if r["vel_meas"] is not None])
    tm_sp = np.array([r["t"] for r in series if r["vel_meas"] is not None])

    t_last = t[-1]
    title = (f"{trial.group}  {trial.scenario}  seed={trial.seed}  ->  "
             f"{trial.status}  (v_end={trial.end_speed_m_s:.3f} m/s, "
             f"v_tol={L.V_TOL_M_S} m/s)")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax1, ax2, ax3 = axes

    # 1) true height
    ax1.plot(t, h, "k-", lw=1.2, label="Hoehe h(t)")
    ax1.set_ylabel("h [m]")
    ax1.set_title(title, fontsize=11)
    ax1.grid(True, alpha=0.3); ax1.legend(loc="best", fontsize=8)

    # 2) vertical velocity: true / measured / estimate
    ax2.plot(t, vz_true, "k-", lw=1.6, label="wahrer vz")
    ax2.plot(t, vz_est, "tab:blue", lw=1.6, label="Schaetzung vz (Filter)")
    ax2.scatter(tm_valid, vm_true, s=14, color="tab:orange", alpha=0.55,
                label="Messung vz (gueltig)")
    ax2.axhline(0.0, color="gray", ls=":", lw=0.8)
    ax2.axhline(-sc.descent_speed_m_s, color="green", ls="--", lw=0.8,
                label=f"Soll-Sinkrate -{sc.descent_speed_m_s} m/s")
    ax2.set_ylabel("vz [m/s]")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) speed magnitude vs soft-landing limit
    ax3.plot(t, sp_true, "k-", lw=1.6, label="|v| wahr")
    ax3.plot(t, sp_est, "tab:blue", lw=1.6, label="|v| Schaetzung")
    ax3.scatter(tm_sp, sp_meas, s=14, color="tab:orange", alpha=0.55,
                label="|v| Messung")
    ax3.axhline(L.V_TOL_M_S, color="red", ls="--", lw=1.2,
                label=f"v_tol = {L.V_TOL_M_S} m/s")
    ax3.set_ylabel("|v| [m/s]")
    ax3.set_xlabel("t [s]")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ground contact marker + status label (all panels)
    for ax in axes:
        ax.axvline(t_last, color="red", ls=":", lw=1.4)
        ax.annotate("Bodenkontakt", xy=(t_last, ax.get_ylim()[0]),
                    xytext=(t_last + 0.3, ax.get_ylim()[0]),
                    color="red", fontsize=8, va="bottom")
    if trial.status == L.TOO_FAST:
        ax3.annotate(f"TOO_FAST\nv_end={trial.end_speed_m_s:.3f} > {L.V_TOL_M_S}",
                     xy=(t_last, trial.end_speed_m_s),
                     xytext=(0.4 * t_last, 0.9),
                     arrowprops=dict(arrowstyle="->", color="red"),
                     color="red", fontsize=10,
                     bbox=dict(facecolor="white", edgecolor="red", alpha=0.9))

    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def main(outdir: str = "output/sensor_diagnostics", show: bool = False,
         seeds=SE.DEFAULT_SEEDS, scenarios=None):
    # Same scenarios + seeds as the experiment (only estimation mode fails).
    scenarios = scenarios if scenarios is not None else SE.COMPARISON_SCENARIOS
    trials = SE.run_repeated(scenarios=scenarios, seeds=seeds,
                             groups=("estimation",))
    failures = find_failures(trials)

    rows = []
    for f in failures:
        sc = scenario_by_name(f.scenario, scenarios)
        cfg = replace(SE.DEFAULT_SENSOR_CFG, seed=f.seed)
        trial, series = SE.run_group_trial_with_series(
            "estimation", f.scenario, sc, cfg)
        # consistency: the re-run (with recording) must reproduce the summary.
        assert trial.status == f.status
        assert abs(trial.end_speed_m_s - f.end_speed_m_s) < 1e-9
        metrics = lag_metrics(trial, series)
        row = failure_row(trial, metrics)
        rows.append(row)
        plot_speed_diagnostics(
            trial, series, sc,
            path=os.path.join(outdir, "failures",
                              f"estimation__{f.scenario}__seed{f.seed}.png"),
            show=show)

    if rows:
        csv_path = write_failures_csv(rows, os.path.join(outdir, "failures.csv"))
    else:
        csv_path = None

    print(f"Versuche (nur estimation): {len(trials)}  |  Fehlschlaege: {len(failures)}")
    print(f"{'scenario':<10}{'seed':>6}{'v_end':>8}{'vz_true':>9}{'vz_est':>9}"
          f"{'delta':>8}{'max|dev|':>9}{'t_max':>7}{'|dev| letzte 2s':>16}")
    for r in rows:
        print(f"{r['scenario']:<10}{r['seed']:>6}{r['v_end_m_s']:>8.3f}"
              f"{r['vz_true_end_m_s']:>9.3f}{r['vz_est_end_m_s']:>9.3f}"
              f"{r['vz_end_delta_m_s']:>8.3f}{r['max_abs_dev_m_s']:>9.3f}"
              f"{r['t_max_dev_s']:>7.2f}{r['mean_abs_dev_last_2s_m_s']:>16.3f}")
    if csv_path:
        print(f"CSV:   {csv_path}")
        print(f"Plots: {os.path.join(outdir, 'failures')}/")
    return rows


if __name__ == "__main__":
    main()