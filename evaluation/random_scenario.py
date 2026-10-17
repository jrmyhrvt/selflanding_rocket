"""
Random-scenario mode for the UNIFIED 3D flight simulation.

Purpose: run MANY randomly sampled, physically valid initial conditions through
the SAME 3D flight physics and the SAME automatic controller, assess each run
with the SAME landing criteria, and report statistics + reproducible artifacts.

Reused building blocks (no second physics/control implementation):
    simulation.flight.run_flight / FlightConfig   <- 3D physics + TVC servos
    simulation.flight.model_control               <- existing automatic control
    evaluation.landing.evaluate_landing           <- landing criteria
    sensors.observed.SensorConfig                 <- sensor parameters
    sensors.observed_flight.ObservedFlightController <- meas./estimate layer
    estimator.base.LowPassEstimator               <- optional estimate

Target point (convention, documented): the commanded landing point is the FIXED
point `target_center_xy` at ground level (a real landing pad), so the random
horizontal start offset (`offset_radius_m`) is a genuine POSITION error the
controller must remove while descending. The single source of truth is
`target_point(cfg, ic)`, used by BOTH the controller (through
`FlightConfig.target_pos`) and the landing evaluation, so the two can never
disagree. The controller's outer POSITION loop (`kp_xy`, see
`simulation/flight.py::_horizontal_velocity_command`) steers the vehicle over
that point, on top of the existing lateral VELOCITY damping. With `kp_xy = 0` the
controller only damps the horizontal velocity (legacy behaviour), so it cannot
remove a position offset and the run tends to end OFF_TARGET. Set
`target_from_start = True` for the old "target = start position" convention (then
`offset_radius_m` becomes inert, because the flat-earth dynamics are translation
invariant).

Statuses (mutually exclusive):
    LANDED       soft touchdown (v <= v_tol) on target (dist <= dist_tol)
    TOO_FAST     touched down but v > v_tol
    OFF_TARGET   touched down but horizontal dist > dist_tol
    CONTROL_LOSS attitude envelope exceeded (max tilt / max rate > tolerance)
    TIMEOUT      run ended without ground contact (simulation limit)
    NO_DATA      empty run

A run that ends airborne is NEVER a success. A timeout/control-loss is never a
success either.

Learning model only. NOT for real hardware.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import csv
import json
import os

import numpy as np

from config import units
from physics.rotation import rotate_vector, normalize
from simulation.flight import FlightConfig, run_flight, model_control
from sensors.observed import SensorConfig
from sensors.observed_flight import ObservedFlightController
from estimator.base import LowPassEstimator
from evaluation import landing as L


# ---------------------------------------------------------------- statuses
LANDED = "LANDED"
TOO_FAST = "TOO_FAST"
OFF_TARGET = "OFF_TARGET"
CONTROL_LOSS = "CONTROL_LOSS"
TIMEOUT = "TIMEOUT"
NO_DATA = "NO_DATA"

STATUSES = (LANDED, TOO_FAST, OFF_TARGET, CONTROL_LOSS, TIMEOUT, NO_DATA)

# Guard: keep the number of integration steps per run bounded.
MAX_STEPS = 200000


@dataclass
class RandomScenarioConfig:
    """Documented, configurable settings for the random-scenario mode.

    All ranges are inclusive-ish uniform ranges (SI units). Every random draw
    comes from one `numpy.random.default_rng`, so a fixed `seed` reproduces the
    whole series exactly; `seed=None` picks a random seed that is reported.
    """
    n_runs: int = 100
    seed: Optional[int] = None

    # --- random initial position / height -------------------------------
    height_range_m: Tuple[float, float] = (5.0, 20.0)
    offset_radius_m: float = 1.5          # horizontal offset in a disc around target_center_xy
    target_center_xy: Tuple[float, float] = (0.0, 0.0)
    # Target definition. False (default): the target is the FIXED point
    # `target_center_xy` (a real landing pad), so the random start offset is a
    # genuine horizontal position error the controller must remove - this is the
    # only way `offset_radius_m` and the position loop are meaningful. True: the
    # target is the horizontal START position (old behaviour; then the start
    # offset is NOT an error and `offset_radius_m` has no effect at all because
    # the flat-earth dynamics are translation invariant).
    target_from_start: bool = False

    # --- random initial velocity (inertial frame) -----------------------
    v_vertical_range_m_s: Tuple[float, float] = (-1.5, 0.5)
    v_horizontal_range_m_s: Tuple[float, float] = (-0.5, 0.5)   # per axis

    # --- random initial attitude (tilt about a random horizontal axis) --
    max_tilt_deg: float = 6.0
    # --- random initial body angular velocity (per axis) ----------------
    max_omega_deg_s: float = 20.0

    # --- vehicle / model (forwarded to FlightConfig) -------------------
    dry_mass_kg: float = 50.0
    fuel_kg: float = 10.0
    I_diag: Tuple[float, float, float] = (8.0, 8.0, 3.0)
    max_thrust_N: float = 4000.0
    gimbal_max_rad: float = 0.35
    gimbal_offset_m: float = 0.30

    # --- simulation -----------------------------------------------------
    dt: float = 0.02
    tf: float = 60.0
    ground_z: float = 0.0

    # --- existing automatic controller (unchanged) ---------------------
    # NOTE: the 3D controller tracks a CONSTANT desired sink rate `vz_des_m_s`
    # (no terminal flush). For a soft touchdown under the default v_tol = 0.5 m/s
    # the commanded sink rate must be BELOW that tolerance, hence -0.4 here. The
    # interactive app keeps its own -2.0 (fast descent demo).
    vz_des_m_s: float = -0.4
    kv_vh: float = 0.6
    max_tilt_control_deg: float = 20.0
    kp_att: float = 4.0
    kd_att: float = 3.5
    kv_vz: float = 1.0
    # Outer horizontal POSITION loop of the 3D controller (see
    # simulation/flight.py::_horizontal_velocity_command). kp_xy turns the
    # horizontal offset from the target into a target-ward velocity setpoint,
    # clamped to vh_max_m_s. kp_xy = 0 => legacy velocity-only behaviour.
    kp_xy: float = 0.4
    vh_max_m_s: float = 2.0

    # --- sensor / estimation layer -------------------------------------
    sensor_enabled: bool = True
    pos_std_m: float = 1.0
    vel_std_m_s: float = 0.3
    gyro_std_rad_s: float = 0.0
    dropout_prob: float = 0.0
    estimator_enabled: bool = False
    estimator_alpha: float = 0.25

    # --- assessment thresholds -----------------------------------------
    v_tol_m_s: float = L.V_TOL_M_S
    h_tol_m: float = L.H_TOL_M
    dist_tol_m: float = L.DIST_TOL_M
    attitude_dev_tol_deg: float = 60.0    # control-loss: max tilt from upright
    omega_tol_deg_s: float = 720.0        # control-loss: max |omega_b|

    def validate(self) -> None:
        if self.n_runs < 0:
            raise ValueError("n_runs muss >= 0 sein")
        if self.dt <= 0.0:
            raise ValueError("Zeitschritt dt muss > 0 sein")
        if self.dt > 1.0:
            raise ValueError("Zeitschritt dt ist zu gross (max 1.0 s)")
        if self.tf <= 0.0:
            raise ValueError("tf muss > 0 sein")
        if self.tf / self.dt > MAX_STEPS:
            raise ValueError(
                f"zu viele Schritte ({int(self.tf / self.dt)}); dt vergroessern")
        if self.dry_mass_kg <= 0.0:
            raise ValueError("Trockenmasse muss > 0 sein")
        if self.fuel_kg < 0.0:
            raise ValueError("Treibstoffmasse darf nicht negativ sein")
        if self.max_thrust_N <= 0.0:
            raise ValueError("max_thrust_N muss > 0 sein")
        if self.gimbal_max_rad <= 0.0:
            raise ValueError("gimbal_max_rad muss > 0 sein")
        self._range("height_range_m", self.height_range_m, lo_min=0.0)
        self._range("v_vertical_range_m_s", self.v_vertical_range_m_s)
        self._range("v_horizontal_range_m_s", self.v_horizontal_range_m_s)
        if self.offset_radius_m < 0.0:
            raise ValueError("offset_radius_m muss >= 0 sein")
        if not (0.0 <= self.max_tilt_deg <= 180.0):
            raise ValueError("max_tilt_deg muss in [0, 180] liegen")
        if self.max_omega_deg_s < 0.0:
            raise ValueError("max_omega_deg_s muss >= 0 sein")
        if self.v_tol_m_s < 0.0 or self.h_tol_m < 0.0 or self.dist_tol_m < 0.0:
            raise ValueError("Landetoleranzen muessen >= 0 sein")
        if not (0.0 < self.attitude_dev_tol_deg <= 180.0):
            raise ValueError("attitude_dev_tol_deg muss in (0, 180] liegen")
        if self.omega_tol_deg_s <= 0.0:
            raise ValueError("omega_tol_deg_s muss > 0 sein")
        if self.kp_xy < 0.0:
            raise ValueError("kp_xy muss >= 0 sein")
        if self.vh_max_m_s < 0.0:
            raise ValueError("vh_max_m_s muss >= 0 sein")
        SensorConfig(pos_std_m=self.pos_std_m, vel_std_m_s=self.vel_std_m_s,
                     dropout_prob=self.dropout_prob, seed=0).validate()
        if self.gyro_std_rad_s < 0.0:
            raise ValueError("gyro_std_rad_s muss >= 0 sein")
        if not (0.0 < self.estimator_alpha <= 1.0):
            raise ValueError("estimator_alpha muss in (0, 1] liegen")

    @staticmethod
    def _range(name, rng, lo_min=None):
        lo, hi = float(rng[0]), float(rng[1])
        if lo > hi:
            raise ValueError(f"{name}: untere Grenze > obere Grenze")
        if lo_min is not None and lo < lo_min:
            raise ValueError(f"{name}: Werte muessen >= {lo_min} sein")


# ---------------------------------------------------------------- sampling
def sample_initial_state(rng: np.random.Generator, cfg: RandomScenarioConfig) -> dict:
    """Draw ONE random, physically valid initial state.

    Guarantees (also asserted below): finite position/velocity, a finite UNIT
    quaternion (no NaN/inf, no zero quaternion). Reproducible from `rng`.
    """
    # position: height uniform, horizontal offset uniform in a disc.
    z0 = float(rng.uniform(*cfg.height_range_m))
    r = cfg.offset_radius_m * np.sqrt(float(rng.uniform(0.0, 1.0)))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    x0 = float(cfg.target_center_xy[0]) + r * np.cos(theta)
    y0 = float(cfg.target_center_xy[1]) + r * np.sin(theta)
    pos = np.array([x0, y0, z0], dtype=float)

    # velocity: vertical + horizontal (per axis), inertial frame.
    vz = float(rng.uniform(*cfg.v_vertical_range_m_s))
    vx = float(rng.uniform(*cfg.v_horizontal_range_m_s))
    vy = float(rng.uniform(*cfg.v_horizontal_range_m_s))
    vel = np.array([vx, vy, vz], dtype=float)

    # attitude: tilt of `tilt` about a random horizontal axis (z-component 0).
    tilt = float(np.deg2rad(rng.uniform(0.0, cfg.max_tilt_deg)))
    phi = float(rng.uniform(0.0, 2.0 * np.pi))
    axis = np.array([np.cos(phi), np.sin(phi), 0.0], dtype=float)
    q = normalize(np.concatenate([[np.cos(tilt / 2.0)], np.sin(tilt / 2.0) * axis]))

    # body angular velocity, per axis.
    wmax = np.deg2rad(cfg.max_omega_deg_s)
    omega = np.array([float(rng.uniform(-wmax, wmax)) for _ in range(3)], dtype=float)

    state = {"pos": pos, "vel": vel, "q": q, "omega_b": omega}
    _assert_valid_initial_state(state)
    return state


def _assert_valid_initial_state(state: dict) -> None:
    for key in ("pos", "vel", "q", "omega_b"):
        v = np.asarray(state[key], dtype=float)
        if not np.all(np.isfinite(v)):
            raise ValueError(f"ungueltiger Anfangszustand: {key} nicht endlich")
    q = np.asarray(state["q"], dtype=float)
    if abs(float(np.linalg.norm(q)) - 1.0) > 1e-9:
        raise ValueError("ungueltiger Anfangszustand: Quaternion nicht normiert")


def initial_tilt_deg(state: dict) -> float:
    """Angle between the body long axis (+Z) and the world up axis, in degrees."""
    up_b = rotate_vector(np.asarray(state["q"], float), np.array([0.0, 0.0, 1.0]))
    z = float(np.clip(up_b[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(z)))


def target_point(cfg: RandomScenarioConfig, ic: dict) -> np.ndarray:
    """The horizontal landing TARGET for one sampled initial state.

    Single source of truth: the SAME helper feeds BOTH the controller
    (`FlightConfig.target_pos`) and the landing evaluation (`evaluate_landing`),
    so the target the controller flies to can never disagree with the target the
    run is scored against.

    - `target_from_start = False` (default): the fixed landing pad
      `(target_center_xy, ground_z)`. The random start offset is then a real
      horizontal position error.
    - `target_from_start = True`: the horizontal START position
      `(x0, y0, ground_z)` (old behaviour).
    """
    if cfg.target_from_start:
        return np.array([float(ic["pos"][0]), float(ic["pos"][1]),
                         float(cfg.ground_z)])
    return np.array([float(cfg.target_center_xy[0]),
                     float(cfg.target_center_xy[1]), float(cfg.ground_z)])


def build_flight_config(cfg: RandomScenarioConfig, ic: dict) -> FlightConfig:
    """Assemble the FlightConfig for one sampled initial state.

    Reuses the EXISTING FlightConfig/run_flight; only the initial state and the
    documented control/vehicle parameters are filled in. The target point comes
    from `target_point` (same point used for the evaluation).
    """
    target = target_point(cfg, ic)
    return FlightConfig(
        t0=0.0, tf=cfg.tf, dt=cfg.dt, gravity=True, stop_at_ground=True,
        ground_z=cfg.ground_z,
        start_pos=np.asarray(ic["pos"], float).copy(),
        start_vel=np.asarray(ic["vel"], float).copy(),
        start_q=np.asarray(ic["q"], float).copy(),
        start_omega_b=np.asarray(ic["omega_b"], float).copy(),
        dry_mass_kg=cfg.dry_mass_kg, fuel_kg=cfg.fuel_kg,
        I_diag=np.asarray(cfg.I_diag, float),
        max_thrust_N=cfg.max_thrust_N, gimbal_max_rad=cfg.gimbal_max_rad,
        gimbal_offset_m=cfg.gimbal_offset_m,
        use_controller=True, vz_des_m_s=cfg.vz_des_m_s, kv_vh=cfg.kv_vh,
        max_tilt_deg=cfg.max_tilt_control_deg, kp_att=cfg.kp_att,
        kd_att=cfg.kd_att, kv_vz=cfg.kv_vz,
        vh_des_m_s=np.zeros(2),
        target_pos=target, kp_xy=cfg.kp_xy, vh_max_m_s=cfg.vh_max_m_s,
    )


# ---------------------------------------------------------------- trial
@dataclass
class RandomTrial:
    run: int
    seed: int
    # initial conditions (SI)
    pos_x0_m: float
    pos_y0_m: float
    pos_z0_m: float
    vel_x0_m_s: float
    vel_y0_m_s: float
    vel_z0_m_s: float
    tilt0_deg: float
    omega_x0_rad_s: float
    omega_y0_rad_s: float
    omega_z0_rad_s: float
    target_x_m: float
    target_y_m: float
    # result / measurements
    status: str
    success: bool
    failure_cause: str
    contact: bool
    n_steps: int
    duration_s: float
    end_height_m: float
    v_total_m_s: float
    v_vertical_m_s: float
    v_horizontal_m_s: float
    target_distance_m: float
    max_attitude_dev_deg: float
    max_omega_deg_s: float
    # sensor / estimate telemetry
    sensor_enabled: bool
    est_pos_rms_m: float
    est_vel_rms_m_s: float
    samples_missing: int


def _max_attitude_dev_deg(history) -> float:
    best = 0.0
    for s in history:
        up_b = rotate_vector(np.asarray(s.q, float), np.array([0.0, 0.0, 1.0]))
        z = float(np.clip(up_b[2], -1.0, 1.0))
        best = max(best, float(np.degrees(np.arccos(z))))
    return best


def run_random_trial(cfg: RandomScenarioConfig, run: int,
                     rng: np.random.Generator, seed: int) -> RandomTrial:
    """Sample ONE initial state, run it, assess it and classify the outcome."""
    cfg.validate()
    ic = sample_initial_state(rng, cfg)
    target_pos = target_point(cfg, ic)
    flight = build_flight_config(cfg, ic)

    controller = None
    if cfg.sensor_enabled:
        sensor_seed = int(rng.integers(0, 2 ** 31 - 1))
        sensor_cfg = SensorConfig(pos_std_m=cfg.pos_std_m,
                                  vel_std_m_s=cfg.vel_std_m_s,
                                  dropout_prob=cfg.dropout_prob,
                                  seed=sensor_seed)
        est = (LowPassEstimator(cfg.estimator_alpha, cfg.estimator_alpha)
               if cfg.estimator_enabled else None)
        controller = ObservedFlightController(
            model_control, sensor_cfg, gyro_std_rad_s=cfg.gyro_std_rad_s,
            estimator=est)

    hist = run_flight(flight, controller=controller)
    tilt0 = initial_tilt_deg(ic)

    if not hist:
        return RandomTrial(
            run=run, seed=seed, pos_x0_m=float(ic["pos"][0]),
            pos_y0_m=float(ic["pos"][1]), pos_z0_m=float(ic["pos"][2]),
            vel_x0_m_s=float(ic["vel"][0]), vel_y0_m_s=float(ic["vel"][1]),
            vel_z0_m_s=float(ic["vel"][2]), tilt0_deg=tilt0,
            omega_x0_rad_s=float(ic["omega_b"][0]),
            omega_y0_rad_s=float(ic["omega_b"][1]),
            omega_z0_rad_s=float(ic["omega_b"][2]),
            target_x_m=float(target_pos[0]), target_y_m=float(target_pos[1]),
            status=NO_DATA, success=False, failure_cause=NO_DATA, contact=False,
            n_steps=0, duration_s=float("nan"), end_height_m=float("nan"),
            v_total_m_s=float("nan"), v_vertical_m_s=float("nan"),
            v_horizontal_m_s=float("nan"), target_distance_m=float("nan"),
            max_attitude_dev_deg=float("nan"), max_omega_deg_s=float("nan"),
            sensor_enabled=bool(cfg.sensor_enabled), est_pos_rms_m=float("nan"),
            est_vel_rms_m_s=float("nan"), samples_missing=0)

    res = L.evaluate_landing(hist, target_pos, ground_z=cfg.ground_z,
                             v_tol=cfg.v_tol_m_s, h_tol=cfg.h_tol_m,
                             dist_tol=cfg.dist_tol_m)
    max_dev = _max_attitude_dev_deg(hist)
    max_omega = max(float(np.linalg.norm(np.asarray(s.omega_b, float)))
                    for s in hist)
    max_omega_deg = float(np.degrees(max_omega))

    control_loss = (max_dev > cfg.attitude_dev_tol_deg
                    or max_omega_deg > cfg.omega_tol_deg_s)
    if control_loss:
        status = CONTROL_LOSS
    elif res["status"] == L.LANDED:
        status = LANDED
    elif res["status"] == L.TOO_FAST:
        status = TOO_FAST
    elif res["contact"]:
        status = OFF_TARGET
    else:
        status = TIMEOUT
    success = (status == LANDED)

    last = hist[-1]
    if controller is not None:
        epos, evel = controller.rms_errors()
        missing = int(controller.samples_missing)
    else:
        epos = evel = float("nan")
        missing = 0

    return RandomTrial(
        run=run, seed=seed,
        pos_x0_m=float(ic["pos"][0]), pos_y0_m=float(ic["pos"][1]),
        pos_z0_m=float(ic["pos"][2]),
        vel_x0_m_s=float(ic["vel"][0]), vel_y0_m_s=float(ic["vel"][1]),
        vel_z0_m_s=float(ic["vel"][2]), tilt0_deg=tilt0,
        omega_x0_rad_s=float(ic["omega_b"][0]),
        omega_y0_rad_s=float(ic["omega_b"][1]),
        omega_z0_rad_s=float(ic["omega_b"][2]),
        target_x_m=float(target_pos[0]), target_y_m=float(target_pos[1]),
        status=status, success=success,
        failure_cause=("" if success else status),
        contact=bool(res.get("contact", False)),
        n_steps=len(hist), duration_s=float(last.t),
        end_height_m=float(last.height_m),
        v_total_m_s=float(np.linalg.norm(last.vel)),
        v_vertical_m_s=float(last.vel[2]),
        v_horizontal_m_s=float(np.hypot(last.vel[0], last.vel[1])),
        target_distance_m=float(res["dist_xy"]),
        max_attitude_dev_deg=max_dev, max_omega_deg_s=max_omega_deg,
        sensor_enabled=bool(cfg.sensor_enabled),
        est_pos_rms_m=float(epos) if epos is not None else float("nan"),
        est_vel_rms_m_s=float(evel) if evel is not None else float("nan"),
        samples_missing=missing)


def resolved_seed(cfg: RandomScenarioConfig) -> int:
    """The seed actually used: the configured one, or a fresh random seed."""
    if cfg.seed is not None:
        return int(cfg.seed)
    return int(np.random.SeedSequence().entropy)


def run_random_scenarios(cfg: RandomScenarioConfig,
                         seed: Optional[int] = None) -> list:
    """Run the full series. Reproducible for a fixed (cfg ranges, seed)."""
    cfg.validate()
    base_seed = resolved_seed(cfg) if seed is None else int(seed)
    rng = np.random.default_rng(base_seed)
    return [run_random_trial(cfg, i, rng, base_seed) for i in range(cfg.n_runs)]


# ---------------------------------------------------------------- statistics
def _stats(values) -> dict:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(vals, dtype=float)
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "median": float(np.median(arr)), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


def summarize(trials, cfg: RandomScenarioConfig, seed: int) -> dict:
    """Aggregate statistics over all runs (failed runs are always counted)."""
    n = len(trials)
    landed = sum(1 for t in trials if t.success)
    status_counts = {st: sum(1 for t in trials if t.status == st)
                     for st in STATUSES}
    contact = [t for t in trials if t.contact]
    failure_causes = {st: status_counts[st] for st in STATUSES if st != LANDED}
    return {
        "config": asdict(cfg),
        "seed": int(seed),
        "n_runs": n,
        "success": {
            "landed": landed,
            "failed": n - landed,
            "success_rate": (landed / n) if n else 0.0,
        },
        "status_counts": status_counts,
        "failure_causes": failure_causes,
        "touchdown_speed_m_s": _stats([t.v_total_m_s for t in contact]),
        "target_distance_m": _stats([t.target_distance_m for t in contact]),
        "max_attitude_dev_deg": _stats([t.max_attitude_dev_deg for t in trials]),
        "max_omega_deg_s": _stats([t.max_omega_deg_s for t in trials]),
        "duration_s": _stats([t.duration_s for t in trials]),
        "sensor": {
            "enabled": bool(cfg.sensor_enabled),
            "mean_est_pos_rms_m": _stats([t.est_pos_rms_m for t in trials])["mean"],
            "mean_est_vel_rms_m_s": _stats([t.est_vel_rms_m_s for t in trials])["mean"],
            "mean_samples_missing": _stats(
                [float(t.samples_missing) for t in trials])["mean"],
        },
        "thresholds": {
            "v_tol_m_s": cfg.v_tol_m_s, "h_tol_m": cfg.h_tol_m,
            "dist_tol_m": cfg.dist_tol_m,
            "attitude_dev_tol_deg": cfg.attitude_dev_tol_deg,
            "omega_tol_deg_s": cfg.omega_tol_deg_s,
        },
    }


# ---------------------------------------------------------------- reporting
def format_table(trials) -> str:
    header = (f"{'run':>4}{'status':>12}{'v_end':>8}{'dist':>7}"
              f"{'tilt_max':>9}{'om_max':>9}{'t_end':>7}{'steps':>7}")
    lines = [header, "-" * len(header)]
    for t in trials:
        lines.append(
            f"{t.run:>4}{t.status:>12}{t.v_total_m_s:>8.2f}"
            f"{t.target_distance_m:>7.2f}{t.max_attitude_dev_deg:>9.1f}"
            f"{t.max_omega_deg_s:>9.1f}{t.duration_s:>7.2f}{t.n_steps:>7d}")
    return "\n".join(lines)


def write_results_csv(trials, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(asdict(trials[0]).keys()) if trials else [
        "run", "seed", "status", "success"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for t in trials:
            writer.writerow(asdict(t))
    return path


def write_summary_json(summary: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    return path


def plot_distributions(trials, summary, path: str = None, show: bool = False):
    import matplotlib
    if path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

    # 1) success vs failed
    s = summary["success"]
    ax1.bar(["LANDED", "FAILED"], [s["landed"], s["failed"]],
            color=["tab:green", "tab:red"], alpha=0.75)
    ax1.set_title(f"Erfolg: {s['landed']}/{summary['n_runs']} "
                  f"({s['success_rate']*100:.0f}%)")
    ax1.set_ylabel("Anzahl"); ax1.grid(True, axis="y", alpha=0.3)

    # 2) status distribution (including failure causes)
    counts = summary["status_counts"]
    labels = [st for st in STATUSES if counts.get(st, 0) > 0]
    ax2.bar(labels, [counts[st] for st in labels], color="tab:blue", alpha=0.75)
    ax2.set_title("Statusverteilung / Fehlerursachen")
    ax2.set_ylabel("Anzahl"); ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, axis="y", alpha=0.3)

    def _hist(ax, values, tol, title, xlabel, color):
        vals = [v for v in values if v is not None and np.isfinite(v)]
        if vals:
            ax.hist(vals, bins=min(20, max(3, len(vals))), color=color,
                    alpha=0.75, edgecolor="k", linewidth=0.4)
        if tol is not None:
            ax.axvline(tol, color="r", ls="--", lw=1.2, label=f"tol = {tol}")
            ax.legend(fontsize=8)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("Anzahl")
        ax.grid(True, alpha=0.3)

    contact = [t for t in trials if t.contact]
    _hist(ax3, [t.v_total_m_s for t in contact], summary["thresholds"]["v_tol_m_s"],
          "Aufsetzgeschwindigkeit", "v_end [m/s]", "tab:blue")
    _hist(ax4, [t.target_distance_m for t in contact],
          summary["thresholds"]["dist_tol_m"],
          "Zielabstand", "dist [m]", "tab:purple")
    _hist(ax5, [t.max_attitude_dev_deg for t in trials],
          summary["thresholds"]["attitude_dev_tol_deg"],
          "max. Lageabweichung", "tilt [deg]", "tab:orange")
    _hist(ax6, [t.max_omega_deg_s for t in trials],
          summary["thresholds"]["omega_tol_deg_s"],
          "max. |omega_b|", "omega [deg/s]", "tab:red")

    fig.suptitle("Random-Szenario-Modus (3D-Flug, vereinfachtes Lernmodell)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig


def main(outdir: str = "output/random_scenario", cfg: RandomScenarioConfig = None,
         seed: Optional[int] = None, show: bool = False):
    cfg = cfg or RandomScenarioConfig()
    trials = run_random_scenarios(cfg, seed=seed)
    base_seed = resolved_seed(cfg) if seed is None else int(seed)
    summary = summarize(trials, cfg, base_seed)

    print(format_table(trials))
    print("=" * 72)
    s = summary["success"]
    print(f"Seed: {base_seed}  |  Laeufe: {summary['n_runs']}  |  "
          f"LANDED: {s['landed']}  fehlgeschlagen: {s['failed']}  "
          f"({s['success_rate']*100:.0f}%)")
    print("Status: " + "  ".join(f"{st}={summary['status_counts'][st]}"
                                 for st in STATUSES))
    td = summary["touchdown_speed_m_s"]
    print(f"Aufsetzgeschwindigkeit [m/s]: mean={td['mean']:.2f} "
          f"median={td['median']:.2f} max={td['max']:.2f} (n={td['n']})")
    dd = summary["target_distance_m"]
    print(f"Zielabstand [m]: mean={dd['mean']:.2f} max={dd['max']:.2f} (n={dd['n']})")

    csv_path = write_results_csv(trials, os.path.join(outdir, "random_scenarios.csv"))
    json_path = write_summary_json(summary, os.path.join(outdir, "random_summary.json"))
    png_path = os.path.join(outdir, "random_distributions.png")
    plot_distributions(trials, summary, png_path, show=show)
    print(f"CSV:    {csv_path}")
    print(f"JSON:   {json_path}")
    print(f"Plot:   {png_path}")
    return trials


if __name__ == "__main__":
    main()
