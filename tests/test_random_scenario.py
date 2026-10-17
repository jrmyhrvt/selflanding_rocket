from dataclasses import asdict
import json

import numpy as np
import pytest

from physics.rotation import rotate_vector
from simulation.flight import FlightConfig, run_flight, model_control
from evaluation import landing as L
from evaluation import random_scenario as RS
from evaluation.random_scenario import RandomScenarioConfig


# Small/fast settings so the tests stay quick.
FAST = dict(n_runs=8, tf=8.0, dt=0.02, sensor_enabled=False)


def _cfg(**kw):
    params = dict(FAST)
    params.update(kw)
    return RandomScenarioConfig(**params)


def _same(a, b):
    da, db = asdict(a), asdict(b)
    assert da.keys() == db.keys()
    for k in da:
        va, vb = da[k], db[k]
        if isinstance(va, float) and isinstance(vb, float):
            assert (np.isnan(va) and np.isnan(vb)) or va == vb
        else:
            assert va == vb


# ---------------------------------------------------------------- sampling
def test_sample_initial_state_is_valid():
    cfg = _cfg()
    rng = np.random.default_rng(0)
    for _ in range(50):
        ic = RS.sample_initial_state(rng, cfg)
        for key in ("pos", "vel", "q", "omega_b"):
            assert np.all(np.isfinite(np.asarray(ic[key], float)))
        q = np.asarray(ic["q"], float)
        assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-9


def test_sample_respects_configured_ranges():
    cfg = _cfg(
        height_range_m=(5.0, 20.0), offset_radius_m=3.0,
        v_vertical_range_m_s=(-1.5, 0.5),
        v_horizontal_range_m_s=(-0.5, 0.5),
        max_tilt_deg=6.0, max_omega_deg_s=20.0,
    )
    rng = np.random.default_rng(1234)
    om_max = np.deg2rad(cfg.max_omega_deg_s)
    for _ in range(200):
        ic = RS.sample_initial_state(rng, cfg)
        pos, vel, q, om = ic["pos"], ic["vel"], ic["q"], ic["omega_b"]
        assert cfg.height_range_m[0] <= pos[2] <= cfg.height_range_m[1]
        # horizontal offset is uniform in a disc of radius offset_radius_m
        assert np.hypot(pos[0], pos[1]) <= cfg.offset_radius_m + 1e-9
        assert cfg.v_vertical_range_m_s[0] <= vel[2] <= cfg.v_vertical_range_m_s[1]
        assert cfg.v_horizontal_range_m_s[0] <= vel[0] <= cfg.v_horizontal_range_m_s[1]
        assert cfg.v_horizontal_range_m_s[0] <= vel[1] <= cfg.v_horizontal_range_m_s[1]
        assert np.all(np.abs(om) <= om_max + 1e-12)
        tilt = RS.initial_tilt_deg({"q": q})
        assert -1e-9 <= tilt <= cfg.max_tilt_deg + 1e-6


def test_sample_stays_valid_with_extreme_ranges():
    cfg = _cfg(max_tilt_deg=180.0, max_omega_deg_s=1000.0,
               v_vertical_range_m_s=(-50.0, 50.0),
               v_horizontal_range_m_s=(-50.0, 50.0))
    rng = np.random.default_rng(99)
    for _ in range(50):
        ic = RS.sample_initial_state(rng, cfg)
        for key in ("pos", "vel", "q", "omega_b"):
            assert np.all(np.isfinite(np.asarray(ic[key], float)))
        assert abs(float(np.linalg.norm(ic["q"])) - 1.0) < 1e-9
        assert 0.0 <= RS.initial_tilt_deg(ic) <= 180.0 + 1e-6


def test_initial_tilt_matches_quaternion():
    a = np.deg2rad(10.0)
    q = np.array([np.cos(a / 2.0), np.sin(a / 2.0), 0.0, 0.0])
    assert RS.initial_tilt_deg({"q": q}) == pytest.approx(10.0, abs=1e-6)
    up = rotate_vector(q, np.array([0.0, 0.0, 1.0]))
    assert up[2] == pytest.approx(np.cos(a), abs=1e-9)


# ---------------------------------------------------------------- reproducibility
def test_same_seed_is_reproducible():
    cfg = _cfg(seed=1234, n_runs=6)
    a = RS.run_random_scenarios(cfg)
    b = RS.run_random_scenarios(cfg)
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        _same(ta, tb)


def test_different_seed_changes_initial_conditions():
    cfg = _cfg(n_runs=5)
    a = RS.run_random_scenarios(cfg, seed=1)
    b = RS.run_random_scenarios(cfg, seed=2)
    fa = [(t.pos_x0_m, t.pos_y0_m, t.pos_z0_m, t.vel_x0_m_s, t.vel_y0_m_s,
           t.vel_z0_m_s, t.tilt0_deg) for t in a]
    fb = [(t.pos_x0_m, t.pos_y0_m, t.pos_z0_m, t.vel_x0_m_s, t.vel_y0_m_s,
           t.vel_z0_m_s, t.tilt0_deg) for t in b]
    assert fa != fb


def test_unseeded_series_reports_its_seed():
    cfg = _cfg(n_runs=1)
    trials = RS.run_random_scenarios(cfg)
    summary = RS.summarize(trials, cfg, RS.resolved_seed(cfg))
    assert isinstance(summary["seed"], int)
    assert summary["n_runs"] == 1


# ---------------------------------------------------------------- controller / physics
def test_controller_hook_is_equivalent_to_default():
    cfg = _cfg()
    ic = {"pos": np.array([0.0, 0.0, 10.0]), "vel": np.zeros(3),
          "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega_b": np.zeros(3)}
    flight = RS.build_flight_config(cfg, ic)
    a = run_flight(flight)
    b = run_flight(flight, controller=model_control)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.allclose(sa.pos, sb.pos)
        assert np.allclose(sa.vel, sb.vel)
        assert np.allclose(sa.q, sb.q)


def test_controller_hook_is_invoked_with_the_true_state():
    """The hook is called once per step with the TRUE state and must return a
    3-tuple of (thrust_N, gx_setpoint, gy_setpoint)."""
    calls = []

    def spy(state, m, thr, cfg):
        calls.append((float(state.t), float(m), float(state.pos[2])))
        return float(m * 9.80665), 0.0, 0.0

    flight = RS.build_flight_config(
        _cfg(),
        {"pos": np.array([0.0, 0.0, 10.0]), "vel": np.zeros(3),
         "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega_b": np.zeros(3)})
    hist = run_flight(flight, controller=spy)
    assert len(calls) == len(hist)
    assert calls[0][0] == pytest.approx(flight.t0)
    # the hook saw the true starting mass and height
    assert calls[0][1] == pytest.approx(flight.dry_mass_kg + flight.fuel_kg)
    assert calls[0][2] == pytest.approx(10.0)


def test_model_control_alias_matches_the_internal_default():
    from simulation.flight import _model_control
    assert model_control is _model_control


def test_auto_controller_lands_a_nominal_run():
    """A centred, upright, slow run must be a soft on-target landing."""
    cfg = _cfg(n_runs=1, tf=40.0, dt=0.02, sensor_enabled=False)
    ic = {"pos": np.array([0.0, 0.0, 10.0]), "vel": np.zeros(3),
          "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega_b": np.zeros(3)}
    flight = RS.build_flight_config(cfg, ic)
    hist = run_flight(flight)
    res = L.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]),
                             v_tol=cfg.v_tol_m_s, dist_tol=cfg.dist_tol_m)
    assert res["status"] == L.LANDED


# ---------------------------------------------------------------- target point
def test_target_point_is_the_fixed_landing_pad_by_default():
    cfg = _cfg(target_center_xy=(1.0, -2.0))
    ic = {"pos": np.array([1.5, -2.5, 12.0]), "vel": np.zeros(3),
          "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega_b": np.zeros(3)}
    tgt = RS.target_point(cfg, ic)
    assert np.allclose(tgt, [1.0, -2.0, cfg.ground_z])       # fixed pad, not start


def test_target_point_can_follow_the_start():
    cfg = _cfg(target_from_start=True)
    ic = {"pos": np.array([1.5, -2.5, 12.0]), "vel": np.zeros(3),
          "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega_b": np.zeros(3)}
    assert np.allclose(RS.target_point(cfg, ic), [1.5, -2.5, cfg.ground_z])


def test_start_offset_is_a_real_position_error_by_default():
    """With a fixed target the random start offset must show up as an error."""
    cfg = _cfg(n_runs=8, seed=9, offset_radius_m=2.0, sensor_enabled=False)
    rng = np.random.default_rng(9)
    seen_offset = False
    for _ in range(8):
        ic = RS.sample_initial_state(rng, cfg)
        tgt = RS.target_point(cfg, ic)
        d0 = np.hypot(ic["pos"][0] - tgt[0], ic["pos"][1] - tgt[1])
        if d0 > 1e-9:
            seen_offset = True
    assert seen_offset


def test_target_point_is_shared_by_controller_and_evaluation():
    """The point the controller flies to equals the point the run is scored on.

    Both come from the SAME helper `target_point`, so they cannot disagree.
    """
    cfg = _cfg(n_runs=1, seed=7, sensor_enabled=False)
    rng = np.random.default_rng(7)
    ic = RS.sample_initial_state(rng, cfg)
    flight = RS.build_flight_config(cfg, ic)
    assert np.allclose(flight.target_pos, RS.target_point(cfg, ic))


def test_position_loop_is_active_in_random_trials():
    """Random trials use the outer position loop (target wired into FlightConfig)."""
    cfg = _cfg(n_runs=1, seed=3, sensor_enabled=False)
    assert cfg.kp_xy > 0.0
    ic = RS.sample_initial_state(np.random.default_rng(3), cfg)
    flight = RS.build_flight_config(cfg, ic)
    assert flight.target_pos is not None
    assert flight.kp_xy == cfg.kp_xy
    assert flight.vh_max_m_s == cfg.vh_max_m_s


# ---------------------------------------------------------------- outcomes
def test_status_set_and_landing_eval_consistency():
    cfg = _cfg(n_runs=10, seed=11)
    trials = RS.run_random_scenarios(cfg)
    assert len(trials) == 10
    for t in trials:
        assert t.status in RS.STATUSES
        assert t.success == (t.status == RS.LANDED)
        if not t.success:
            assert t.failure_cause == t.status
        else:
            assert t.failure_cause == ""
        assert t.contact == (t.status in (RS.LANDED, RS.TOO_FAST, RS.OFF_TARGET))
        # a soft on-target landing must satisfy the landing tolerances
        if t.status == RS.LANDED:
            assert t.v_total_m_s <= cfg.v_tol_m_s + 1e-6
            assert t.target_distance_m <= cfg.dist_tol_m + 1e-6


def test_too_fast_is_classified():
    cfg = _cfg(n_runs=8, seed=5, height_range_m=(0.8, 1.5),
               v_vertical_range_m_s=(-4.0, -2.0), tf=10.0, sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    assert all(t.status == RS.TOO_FAST for t in trials)
    assert all(t.v_total_m_s > cfg.v_tol_m_s for t in trials)


def test_off_target_is_classified():
    # Legacy velocity-only controller (kp_xy=0): with no position feedback the
    # initial lateral motion is only damped, so the run DRIFTS away from the
    # start point and is correctly classified OFF_TARGET.
    cfg = _cfg(n_runs=10, seed=11, max_tilt_deg=8.0, kp_xy=0.0,
               v_horizontal_range_m_s=(-1.2, 1.2),
               height_range_m=(5.0, 12.0), tf=40.0, sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    assert any(t.status == RS.OFF_TARGET for t in trials)
    for t in trials:
        if t.status == RS.OFF_TARGET:
            assert t.contact
            assert t.target_distance_m > cfg.dist_tol_m


def test_position_control_removes_off_target_drift():
    """The SAME scenario with the outer position loop lands on target.

    Only `kp_xy` differs - the physics, servos and the rest of the controller are
    unchanged. This is the direct before/after of the horizontal position loop.
    """
    common = dict(n_runs=10, seed=11, max_tilt_deg=8.0,
                  v_horizontal_range_m_s=(-1.2, 1.2),
                  height_range_m=(5.0, 12.0), tf=40.0, sensor_enabled=False)
    legacy = RS.run_random_scenarios(_cfg(kp_xy=0.0, **common))
    controlled = RS.run_random_scenarios(_cfg(kp_xy=0.4, **common))

    def off_target(ts):
        return sum(1 for t in ts if t.status == RS.OFF_TARGET)

    def mean_dist(ts):
        vals = [t.target_distance_m for t in ts
                if np.isfinite(t.target_distance_m)]
        return float(np.mean(vals)) if vals else float("nan")

    assert off_target(controlled) < off_target(legacy)
    assert mean_dist(controlled) < mean_dist(legacy)


def test_control_loss_is_classified():
    cfg = _cfg(n_runs=8, seed=5, omega_tol_deg_s=1.0, tf=10.0, sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    assert all(t.status == RS.CONTROL_LOSS for t in trials)
    assert all(t.max_omega_deg_s > cfg.omega_tol_deg_s for t in trials)


def test_control_loss_takes_precedence_over_too_fast():
    """A run that exceeded the attitude envelope is never counted as a success."""
    cfg = _cfg(n_runs=8, seed=5, omega_tol_deg_s=1.0, tf=10.0, sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    assert all(not t.success for t in trials)
    assert all(t.failure_cause == RS.CONTROL_LOSS for t in trials)


def test_timeout_is_not_a_success():
    cfg = _cfg(n_runs=6, seed=5, height_range_m=(15.0, 20.0), tf=1.0,
               sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    assert all(t.status == RS.TIMEOUT for t in trials)
    assert all(not t.success for t in trials)
    assert all(not t.contact for t in trials)


def test_no_data_status_is_kept_but_not_reached():
    """NO_DATA stays in the status set; the sampler never produces an empty run."""
    assert RS.NO_DATA in RS.STATUSES
    assert L.evaluate_landing([], np.zeros(3))["status"] == L.NO_DATA
    cfg = _cfg(n_runs=4, seed=5)
    trials = RS.run_random_scenarios(cfg)
    assert all(t.status != RS.NO_DATA for t in trials)
    assert all(t.n_steps > 0 for t in trials)


def test_zero_step_flight_returns_empty_history():
    # n_steps <= 0 -> no integration at all -> empty history
    flight = FlightConfig(t0=0.0, tf=0.0, dt=0.02, use_controller=True)
    assert run_flight(flight) == []


# ---------------------------------------------------------------- sensors
def test_sensor_layer_is_applied_when_enabled():
    cfg = _cfg(n_runs=6, seed=3, sensor_enabled=True, dropout_prob=0.4,
               pos_std_m=0.5, vel_std_m_s=0.2)
    trials = RS.run_random_scenarios(cfg)
    for t in trials:
        assert t.sensor_enabled is True
        assert np.isfinite(t.est_pos_rms_m)
        assert np.isfinite(t.est_vel_rms_m_s)
        assert t.samples_missing > 0


def test_no_sensor_metrics_when_disabled():
    cfg = _cfg(n_runs=6, seed=3, sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    for t in trials:
        assert t.sensor_enabled is False
        assert np.isnan(t.est_pos_rms_m)
        assert np.isnan(t.est_vel_rms_m_s)
        assert t.samples_missing == 0


def test_sensor_layer_cannot_change_the_true_physics():
    """The observation layer only changes the COMMAND, never the true state.

    With a controller that ignores what it sees, two very different sensor
    configurations must produce the identical true trajectory.
    """
    from sensors.observed import SensorConfig
    from sensors.observed_flight import ObservedFlightController

    def _const(state, m, thr, cfg):
        return float(m * 9.80665), 0.0, 0.0

    ic = {"pos": np.array([0.0, 0.0, 10.0]), "vel": np.zeros(3),
          "q": np.array([1.0, 0.0, 0.0, 0.0]), "omega_b": np.zeros(3)}
    flight = RS.build_flight_config(_cfg(), ic)
    quiet = ObservedFlightController(
        _const, SensorConfig(pos_std_m=0.0, vel_std_m_s=0.0,
                             dropout_prob=0.0, seed=1))
    noisy = ObservedFlightController(
        _const, SensorConfig(pos_std_m=10.0, vel_std_m_s=5.0,
                             dropout_prob=0.5, seed=2))
    a = run_flight(flight, controller=quiet)
    b = run_flight(flight, controller=noisy)
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert np.allclose(sa.pos, sb.pos)
        assert np.allclose(sa.vel, sb.vel)
        assert np.allclose(sa.q, sb.q)


def test_sampler_stream_is_shared_so_later_runs_shift():
    """Enabling sensors consumes RNG draws (per-run sensor seeds), so the full
    series is not the same series - but run 0 always shares the same IC."""
    cfg_on = _cfg(n_runs=4, seed=3, sensor_enabled=True, pos_std_m=5.0,
                  vel_std_m_s=2.0)
    cfg_off = _cfg(n_runs=4, seed=3, sensor_enabled=False)
    a = RS.run_random_scenarios(cfg_on, seed=3)
    b = RS.run_random_scenarios(cfg_off, seed=3)
    assert a[0].pos_x0_m == b[0].pos_x0_m
    assert a[0].vel_x0_m_s == b[0].vel_x0_m_s
    for t in a:
        assert np.isfinite(t.v_total_m_s)
        assert t.status in RS.STATUSES


def test_estimator_reduces_estimation_error():
    noise = _cfg(n_runs=5, seed=3, sensor_enabled=True, estimator_enabled=False,
                 pos_std_m=1.0, vel_std_m_s=0.5, dropout_prob=0.2)
    est = _cfg(n_runs=5, seed=3, sensor_enabled=True, estimator_enabled=True,
               estimator_alpha=0.1, pos_std_m=1.0, vel_std_m_s=0.5,
               dropout_prob=0.2)
    a = RS.run_random_scenarios(noise, seed=3)
    b = RS.run_random_scenarios(est, seed=3)
    mean_a = np.nanmean([t.est_pos_rms_m for t in a])
    mean_b = np.nanmean([t.est_pos_rms_m for t in b])
    assert mean_b < mean_a


# ---------------------------------------------------------------- count / stats
def test_exactly_100_runs_by_default():
    cfg = RandomScenarioConfig(n_runs=100, seed=7, tf=6.0, dt=0.02,
                               sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    assert len(trials) == 100
    assert [t.run for t in trials] == list(range(100))
    assert all(t.seed == 7 for t in trials)


def test_statistics_summary_structure():
    cfg = _cfg(n_runs=10, seed=1234)
    trials = RS.run_random_scenarios(cfg)
    summ = RS.summarize(trials, cfg, 1234)
    assert summ["n_runs"] == 10
    assert summ["seed"] == 1234
    assert set(summ["status_counts"]) == set(RS.STATUSES)
    assert sum(summ["status_counts"].values()) == 10
    assert 0.0 <= summ["success"]["success_rate"] <= 1.0
    assert summ["success"]["landed"] + summ["success"]["failed"] == 10
    for key in ("touchdown_speed_m_s", "target_distance_m",
                "max_attitude_dev_deg", "max_omega_deg_s", "duration_s"):
        assert set(summ[key]) == {"n", "mean", "median", "std", "min", "max"}
    assert summ["thresholds"]["v_tol_m_s"] == cfg.v_tol_m_s


def test_touchdown_speed_stats_use_contact_runs_only():
    cfg = _cfg(n_runs=6, seed=5, height_range_m=(15.0, 20.0), tf=1.0,
               sensor_enabled=False)
    trials = RS.run_random_scenarios(cfg)
    summ = RS.summarize(trials, cfg, 5)
    # all runs are TIMEOUT (no contact) -> no touchdown-speed samples
    assert summ["touchdown_speed_m_s"]["n"] == 0
    assert np.isnan(summ["touchdown_speed_m_s"]["mean"])


# ---------------------------------------------------------------- outputs
def test_write_results_csv_has_one_row_per_run(tmp_path):
    cfg = _cfg(n_runs=5, seed=21)
    trials = RS.run_random_scenarios(cfg)
    path = tmp_path / "sub" / "random_scenarios.csv"
    RS.write_results_csv(trials, str(path))
    lines = path.read_text().splitlines()
    assert lines[0].split(",") == list(asdict(trials[0]).keys())
    assert len(lines) == 6


def test_main_writes_csv_json_and_plot(tmp_path):
    cfg = _cfg(n_runs=5, seed=21, tf=6.0, dt=0.02)
    outdir = tmp_path / "out"
    trials = RS.main(outdir=str(outdir), cfg=cfg, seed=21)
    assert len(trials) == 5
    csv_p = outdir / "random_scenarios.csv"
    json_p = outdir / "random_summary.json"
    png_p = outdir / "random_distributions.png"
    assert csv_p.exists() and len(csv_p.read_text().splitlines()) == 6
    assert json_p.exists()
    summary = json.loads(json_p.read_text())
    assert summary["seed"] == 21
    assert summary["n_runs"] == 5
    assert set(summary["status_counts"]) == set(RS.STATUSES)
    assert "config" in summary and summary["config"]["n_runs"] == 5
    assert png_p.exists() and png_p.stat().st_size > 0


def test_empty_series_is_handled(tmp_path):
    cfg = _cfg(n_runs=0)
    trials = RS.run_random_scenarios(cfg)
    assert trials == []
    summ = RS.summarize(trials, cfg, 0)
    assert summ["n_runs"] == 0
    assert summ["success"]["success_rate"] == 0.0
    assert summ["touchdown_speed_m_s"]["n"] == 0
    csv_p = tmp_path / "empty.csv"
    RS.write_results_csv(trials, str(csv_p))
    assert csv_p.exists()


def test_plot_distributions_saves_file(tmp_path):
    cfg = _cfg(n_runs=6, seed=21, tf=6.0, dt=0.02)
    trials = RS.run_random_scenarios(cfg)
    summ = RS.summarize(trials, cfg, 21)
    png = tmp_path / "dist.png"
    RS.plot_distributions(trials, summ, str(png))
    assert png.exists() and png.stat().st_size > 0


# ---------------------------------------------------------------- validation
def test_validate_rejects_bad_configs():
    cases = [
        dict(n_runs=-1),
        dict(dt=0.0),
        dict(dt=2.0),
        dict(tf=0.0),
        dict(tf=1e9, dt=0.01),
        dict(dry_mass_kg=0.0),
        dict(fuel_kg=-1.0),
        dict(max_thrust_N=0.0),
        dict(gimbal_max_rad=0.0),
        dict(height_range_m=(20.0, 5.0)),
        dict(v_vertical_range_m_s=(1.0, -1.0)),
        dict(v_horizontal_range_m_s=(1.0, -1.0)),
        dict(height_range_m=(-5.0, 5.0)),
        dict(offset_radius_m=-1.0),
        dict(max_tilt_deg=200.0),
        dict(max_omega_deg_s=-1.0),
        dict(v_tol_m_s=-1.0),
        dict(attitude_dev_tol_deg=0.0),
        dict(omega_tol_deg_s=0.0),
        dict(pos_std_m=-1.0),
        dict(dropout_prob=1.5),
        dict(gyro_std_rad_s=-1.0),
        dict(estimator_alpha=0.0),
        dict(kp_xy=-1.0),
        dict(vh_max_m_s=-1.0),
    ]
    for kwargs in cases:
        with pytest.raises(ValueError):
            _cfg(**kwargs).validate()


def test_extreme_but_valid_config_runs():
    cfg = RandomScenarioConfig(
        n_runs=3, tf=2.0, dt=0.05, sensor_enabled=False,
        height_range_m=(1.0, 3.0), max_tilt_deg=90.0, max_omega_deg_s=500.0,
        v_vertical_range_m_s=(-20.0, 20.0),
        v_horizontal_range_m_s=(-20.0, 20.0),
        max_thrust_N=1e6, attitude_dev_tol_deg=179.0, omega_tol_deg_s=1e6,
    )
    trials = RS.run_random_scenarios(cfg, seed=4)
    assert len(trials) == 3
    for t in trials:
        assert t.status in RS.STATUSES
        assert np.isfinite(t.v_total_m_s)
