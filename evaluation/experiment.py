"""
Experiment / evaluation mode for the simplified autoland learning sim.

Runs several bounded parameter sweeps through the SAME simulator + controller
+ evaluator used by the studio (simulation/scenario.run_scenario and
evaluation/landing.evaluate_landing). NO second physics/control implementation.

Sweeps (documented, deliberately bounded):
    start_height_m   5, 10, 20, 40, 60        [m]
    target_dist_m    0, 1, 2, 4, 8, 12        [m]   target at (d, 0)
    v0_vertical_m_s  0, -1, -2, -5, -10       [m/s]
    dt               0.1, 0.05, 0.02, 0.01, 0.005 [s]
    max_time_s       2, 5, 20, 60, 300        [s]   timeout check:
        the run is cut off after this time; if the rocket is still airborne
        that is FAILED (a timeout must never count as a landing).

Everything else stays at a fixed base scenario. Every trial stores the initial
conditions, end status, simulation duration, end height, vertical/horizontal
touchdown speed, target distance, contact flag and step count.

A timeout (run ended airborne) is FAILED, never a successful landing.

Learning model only (point mass). NOT for real hardware.
"""
from dataclasses import dataclass, asdict, field
import csv
import os
import numpy as np

from simulation.scenario import Scenario, run_scenario
from evaluation import landing as L


# Fixed, documented base scenario (everything not swept).
BASE = dict(
    start_height_m=25.0,
    start_offset_xy=(0.0, 0.0),
    target_xy=(0.0, 0.0),
    v0_vertical_m_s=0.0,
    v0_horizontal_m_s=0.0,
    dt=0.05,
    max_time_s=300.0,
)

# Bounded sweep ranges (chosen to be readable, not exhaustive).
DEFAULT_RANGES = {
    "start_height_m": [5.0, 10.0, 20.0, 40.0, 60.0],
    "target_dist_m": [0.0, 1.0, 2.0, 4.0, 8.0, 12.0],
    "v0_vertical_m_s": [0.0, -1.0, -2.0, -5.0, -10.0],
    "dt": [0.1, 0.05, 0.02, 0.01, 0.005],
    "max_time_s": [2.0, 5.0, 20.0, 60.0, 300.0],
}


@dataclass
class Trial:
    sweep: str
    sweep_value: float
    # result / measurements
    status: str
    success: bool
    contact: bool
    sim_duration_s: float
    end_height_m: float
    v_vertical_m_s: float
    v_horizontal_m_s: float
    v_total_m_s: float
    target_distance_m: float
    n_steps: int
    # scenario / initial conditions
    start_height_m: float
    offset_x_m: float
    offset_y_m: float
    target_x_m: float
    target_y_m: float
    v0_vertical_m_s_init: float
    v0_horizontal_m_s: float
    dt_s: float
    max_time_s: float


def _scenario_for(sweep: str, value: float) -> Scenario:
    params = dict(BASE)
    if sweep == "start_height_m":
        params["start_height_m"] = value
    elif sweep == "target_dist_m":
        params["target_xy"] = (float(value), 0.0)
    elif sweep == "v0_vertical_m_s":
        params["v0_vertical_m_s"] = value
    elif sweep == "dt":
        params["dt"] = value
    elif sweep == "max_time_s":
        params["max_time_s"] = value
    else:
        raise KeyError(f"unknown sweep '{sweep}'")
    return Scenario(**params)


def run_trial(sweep: str, value: float) -> Trial:
    """Run one scenario and measure it. Reuses run_scenario/evaluate_landing."""
    sc = _scenario_for(sweep, value)
    hist = run_scenario(sc)
    res = L.evaluate_landing(hist, sc.target())
    last = hist[-1]
    vxy = float(np.hypot(last.vel[0], last.vel[1]))
    return Trial(
        sweep=sweep,
        sweep_value=float(value),
        status=res["status"],
        success=(res["status"] == L.LANDED),   # timeout (FAILED) is never success
        contact=bool(res["contact"]),
        sim_duration_s=float(last.t),
        end_height_m=float(last.height_m),
        v_vertical_m_s=float(last.vel[2]),
        v_horizontal_m_s=vxy,
        v_total_m_s=float(np.linalg.norm(last.vel)),
        target_distance_m=float(res["dist_xy"]),
        n_steps=len(hist),
        start_height_m=float(sc.start_height_m),
        offset_x_m=float(sc.start_offset_xy[0]),
        offset_y_m=float(sc.start_offset_xy[1]),
        target_x_m=float(sc.target_xy[0]),
        target_y_m=float(sc.target_xy[1]),
        v0_vertical_m_s_init=float(sc.v0_vertical_m_s),
        v0_horizontal_m_s=float(sc.v0_horizontal_m_s),
        dt_s=float(sc.dt),
        max_time_s=float(sc.max_time_s),
    )


def run_experiment(ranges: dict = None) -> list:
    ranges = ranges or DEFAULT_RANGES
    trials = []
    for sweep, values in ranges.items():
        for v in values:
            trials.append(run_trial(sweep, v))
    return trials


def summarize(trials) -> dict:
    """Count successes. A failed/timeout run is never counted as landed."""
    total = len(trials)
    landed = sum(1 for t in trials if t.success)
    return {
        "total": total,
        "landed": landed,
        "failed": total - landed,
        "success_rate": (landed / total) if total else 0.0,
    }


def group_by_sweep(trials) -> dict:
    groups = {}
    for t in trials:
        groups.setdefault(t.sweep, []).append(t)
    return groups


# ---------------------------------------------------------------- reporting
def format_table(trials) -> str:
    header = (f"{'sweep':<18}{'value':>8}{'status':>10}{'dur[s]':>9}"
              f"{'h_end':>8}{'vz_end':>8}{'vxy_end':>9}{'dist':>8}{'steps':>8}")
    lines = [header, "-" * len(header)]
    for t in trials:
        lines.append(
            f"{t.sweep:<18}{t.sweep_value:>8.3f}{t.status:>10}"
            f"{t.sim_duration_s:>9.2f}{t.end_height_m:>8.2f}"
            f"{t.v_vertical_m_s:>8.2f}{t.v_horizontal_m_s:>9.2f}"
            f"{t.target_distance_m:>8.2f}{t.n_steps:>8d}")
    return "\n".join(lines)


def write_results_csv(trials, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(asdict(trials[0]).keys()) if trials else [
        "sweep", "sweep_value", "status", "success"]
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

    groups = group_by_sweep(trials)
    sweeps = [s for s in DEFAULT_RANGES if s in groups]
    n = max(1, len(sweeps))
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow), squeeze=False)
    axes = axes.ravel()

    for ax, sweep in zip(axes, sweeps):
        rows = sorted(groups[sweep], key=lambda t: t.sweep_value)
        x = [r.sweep_value for r in rows]
        v = [r.v_total_m_s for r in rows]
        ax.plot(x, v, "k.-", lw=1.0, label="v_end (total)")
        ax.axhline(L.V_TOL_M_S, color="r", ls="--", lw=1.0, label="v_tol = 0.5")
        for r in rows:
            ax.scatter([r.sweep_value], [r.v_total_m_s],
                       color=("green" if r.success else "red"),
                       zorder=5, s=45,
                       marker=("o" if r.success else "x"))
        if sweep == "dt":
            ax.set_xscale("log")
        ax.set_title(f"{sweep}  (gruen=LANDED, rot=sonst)", fontsize=10)
        ax.set_xlabel(sweep, fontsize=9)
        ax.set_ylabel("v_end [m/s]", fontsize=9)
        ax.grid(True, alpha=0.3)

        # Secondary axis: simulation duration per run (effort / time to land).
        ax2 = ax.twinx()
        ax2.plot(x, [r.sim_duration_s for r in rows],
                 color="tab:blue", ls=":", lw=1.0)
        ax2.set_ylabel("Dauer [s]", fontsize=8, color="tab:blue")
        ax2.tick_params(axis="y", labelsize=7, colors="tab:blue")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="best")

    for ax in axes[len(sweeps):]:
        ax.axis("off")

    fig.suptitle("Autoland Experiment - vereinfachtes Punktmassen-Modell", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def main(outdir: str = "output/experiment", show: bool = False, ranges: dict = None):
    trials = run_experiment(ranges)
    print(format_table(trials))
    summ = summarize(trials)
    print("-" * 70)
    print(f"Erfolgreiche Landungen: {summ['landed']}/{summ['total']} "
          f"({summ['success_rate']*100:.0f}%), fehlgeschlagen: {summ['failed']}")
    csv_path = write_results_csv(trials, os.path.join(outdir, "experiment_results.csv"))
    png_path = plot_results(trials, os.path.join(outdir, "experiment_results.png"),
                            show=show)
    print(f"CSV:  {csv_path}")
    print(f"Plot: {png_path}")
    return trials


if __name__ == "__main__":
    main()
