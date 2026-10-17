# Random-Szenario-Modus (unified 3D flight)

Learning model only. NOT a real rocket, engine or autopilot.

## Purpose

The interactive app (`flight_sim_app.py`) shows ONE hand-picked scenario. The
random-scenario mode runs a whole **series** of randomly sampled, physically
valid initial conditions through the SAME physics and the SAME automatic
controller, assesses each landing with the existing criteria, and reports
statistics plus reproducible artifacts. No second physics/control implementation
is introduced.

## Reused building blocks (reuse map)

| piece | source | used for |
|---|---|---|
| 3D physics + TVC servos | `simulation/flight.py::run_flight`, `physics/servo.py`, `physics/thrust_model.py` | one run (servos -> thrust force & moment -> translation/rotation) |
| automatic controller | `simulation/flight.py::model_control` (alias of `_model_control`) | the same loops, plus the outer horizontal POSITION loop (`_horizontal_velocity_command`) |
| command hook | `run_flight(cfg, controller=...)` | optional, lets the controller see measured/estimated values |
| landing criteria | `evaluation/landing.py::evaluate_landing` | `LANDED`/`TOO_FAST`/off-target/contact, same tolerances |
| sensor model | `sensors/observed.py::SensorConfig`, `MeasurementModel`; `sensors/posvel.py`, `sensors/imu.py` | position/velocity noise + dropout, body-rate noise |
| estimator | `estimator/base.py::LowPassEstimator` | optional low-pass estimate of pos/vel |
| observation wrapper | `sensors/observed_flight.py::ObservedFlightController` | same rules as `ObservedController`, widened for the 3D controller |
| 3D view | `visualization/flight_view.py` | `--visualize` shows ONE sampled run live |

The point-mass autoland stack (`simulation/scenario.py`, `physics/controller.py`)
is deliberately NOT used: it has no attitude/angular-rate state, so it cannot
express the randomised "Raketenlage und Winkelgeschwindigkeit".

## Data flow

```
RandomScenarioConfig
      |  sample_initial_state(rng, cfg)      -> pos, vel, q (unit), omega_b
      v  build_flight_config(cfg, ic)        -> FlightConfig
   run_flight(flight, controller=?)
      |  per step: model_control -> SERVOS -> thrust force/moment -> translation -> rotation
      |  (optional: ObservedFlightController reads the TRUE state, synthesises
      |   MEASUREMENT/ESTIMATE, and hands the OBSERVED state to model_control)
      v
   evaluate_landing(hist, target_pos) + max attitude deviation + max |omega_b|
      v
   RandomTrial  ->  summarize()  ->  CSV / JSON / PNG
```

The true state is only READ by the observation layer to synthesise a measurement
and to score the estimate; it is never modified. A measurement never steers the
rocket directly.

## Target point (definition + the key finding)

The controller's horizontal setpoint comes from
`simulation/flight.py::_horizontal_velocity_command`. It is the feed-forward
`vh_des_m_s` plus, when a target is given and `kp_xy > 0`, an outer POSITION loop
`kp_xy * (target_xy - pos_xy)` (clamped to `vh_max_m_s`). So besides the sink rate
the controller now also uses the **horizontal target deviation** and the
**lateral velocity**.

The target is the single source of truth `target_point(cfg, ic)`, used by BOTH
the controller (`FlightConfig.target_pos`) and `evaluate_landing`; the two can
never disagree.

- **Default `target_from_start = False`:** the target is the FIXED point
  `target_center_xy` at ground level (a real landing pad). The random start offset
  (`offset_radius_m`) is then a genuine POSITION error, and the position loop must
  remove it while descending.
- **`target_from_start = True`:** the old convention, target = horizontal start
  position.

**Finding (why the default changed):** with the old target = start, the random
offset shifts start AND target together, so the initial horizontal error is
always exactly zero. Because the flat-earth dynamics are translation invariant,
`offset_radius_m` then has **no effect at all** on any metric - verified: 30 runs
with `offset_radius_m` = 0, 1.5, 5.0 gave byte-identical `mean_dist` (1.5040) and
`mean_v` (0.4000). That makes the start offset a no-op and is the wrong setup for
testing position control, so the default is now the fixed landing pad.

Honest note on the earlier "30 LANDED / 70 OFF_TARGET": those runs were NOT
penalised for a start offset (there was none) - they were penalised for the
**drift** produced by the random initial lateral velocity / tilt, which a
velocity-only controller can damp but not undo. The position loop removes that
drift too.

## Descent-rate note (why the default is -0.4)

`model_control` tracks a **constant** desired sink rate `vz_des_m_s`; there is no
terminal flush. Under the default soft-landing tolerance `v_tol = 0.5 m/s` a
constant sink rate above the tolerance would make EVERY run `TOO_FAST`. The
random mode therefore defaults to `vz_des_m_s = -0.4` (below `v_tol`). The
interactive app keeps its own `-2.0` (fast-descent demo). Only a config value
differs - the controller code is identical.

## Randomisation model (validity guarantees)

Everything is drawn from ONE `numpy.random.default_rng`, so a fixed seed
reproduces the whole series exactly. Without a seed a random one is picked and
REPORTED (in the JSON summary and in every CSV row), so any run can be re-run.

| quantity | parameter | sampling |
|---|---|---|
| height | `height_range_m` | uniform |
| horizontal start offset | `offset_radius_m` | uniform in a DISC around `target_center_xy` |
| vertical velocity | `v_vertical_range_m_s` | uniform, per run |
| horizontal velocity | `v_horizontal_range_m_s` | uniform per axis (inertial) |
| attitude | `max_tilt_deg` | tilt about a random horizontal axis, uniform angle |
| body angular velocity | `max_omega_deg_s` | uniform per axis |
| sensor seed | (derived) | one integer per run for the sensor RNG |

`sample_initial_state` asserts (and the tests re-assert): every component is
finite (no `NaN`/`inf`) and `q` is a **unit** quaternion. Invalid configs are
rejected by `RandomScenarioConfig.validate()` (negative ranges, inverted range
bounds, `dt` out of `(0, 1]`, too many steps, negative masses/thresholds, sensor
std < 0, dropout outside `[0, 1]`, ...).

## Sensor / observation layer

`sensors/observed_flight.py::ObservedFlightController` reuses the existing sensor
classes and widens the observation to what the 3D controller reads:

- position / velocity: `MeasurementModel` (`PositionSensor`, `VelocitySensor`,
  dropout), exactly like `sensors/observed.py`.
- body angular rate: `Gyroscope` (`sensors/imu.py`).
- optional `LowPassEstimator` for pos/vel; a missing sample is forwarded as
  `None` so the estimator holds, and without an estimator the last valid sample
  is held (sample-and-hold), same policy as `ObservedController`.

Documented simplification: there is **no attitude sensor** in this learning
model, so the observed state forwards the TRUE orientation `q`. Only position,
velocity and body rate are measured/noisy.

It is injected through the optional, backward-compatible hook:

```python
run_flight(cfg)                          # unchanged default: model_control
run_flight(cfg, controller=model_control)  # identical result (covered by a test)
run_flight(cfg, controller=ObservedFlightController(model_control, sensor_cfg))
```

## Status classification (mutually exclusive)

`evaluate_landing` gives `LANDED` / `TOO_FAST` / `FAILED` / `NO_DATA` plus a
`contact` flag and the target distance. The random mode maps them to a finer,
mutually exclusive set and adds the attitude envelope:

| status | rule | success |
|---|---|---|
| `CONTROL_LOSS` | max tilt > `attitude_dev_tol_deg` OR max `|omega_b|` > `omega_tol_deg_s` | no (takes precedence) |
| `LANDED` | `evaluate_landing` = `LANDED` | **yes** |
| `TOO_FAST` | `evaluate_landing` = `TOO_FAST` (contact, `v > v_tol`) | no |
| `OFF_TARGET` | contact but horizontal dist > `dist_tol` | no |
| `TIMEOUT` | no ground contact by the end (simulation limit) | no |
| `NO_DATA` | empty run | no |

`CONTROL_LOSS` takes precedence: a run that exceeded the attitude envelope is
never counted as a successful landing, even if it later touched down softly. A
run that ends airborne (`TIMEOUT`) is **never** a success.

Thresholds: `v_tol_m_s = L.V_TOL_M_S = 0.5`, `h_tol_m = L.H_TOL_M = 0.05`,
`dist_tol_m = L.DIST_TOL_M = 1.0` (all reused from `evaluation/landing.py`), plus
the two added, documented attitude thresholds `attitude_dev_tol_deg = 60` and
`omega_tol_deg_s = 720`.

## Statistics (`summarize`)

- success rate + `landed`/`failed` counts
- per-status counts = the failure causes
- touchdown-speed distribution (mean/median/std/min/max) over **contact** runs
- target distance, max attitude deviation, max `|omega_b|`, duration
- sensor metrics (estimate RMS, missing samples) when enabled
- the thresholds and the full config used

Failed / timed-out / control-loss runs are always counted; no run is silently
dropped.

## Artifacts (`output/random_scenario/`)

| file | content |
|---|---|
| `random_scenarios.csv` | one row per run: full initial conditions + base `seed` + all result/metric columns |
| `random_summary.json` | statistics + full config + seed + thresholds |
| `random_distributions.png` | success/failure, status counts, and histograms of touchdown speed / target distance / max attitude deviation / max `|omega_b|` with tolerance lines |

## Config reference (defaults)

`RandomScenarioConfig`: `n_runs=100`, `seed=None`, `height_range_m=(5,20)`,
`offset_radius_m=1.5`, `target_center_xy=(0,0)`, `target_from_start=False`,
`v_vertical_range_m_s=(-1.5,0.5)`, `v_horizontal_range_m_s=(-0.5,0.5)`,
`max_tilt_deg=6`, `max_omega_deg_s=20`, `dt=0.02`, `tf=60`, `vz_des_m_s=-0.4`,
`kv_vh=0.6`, `max_tilt_control_deg=20`, `kp_att=4`, `kd_att=3.5`, `kv_vz=1`,
`kp_xy=0.4`, `vh_max_m_s=2.0`,
`sensor_enabled=True`, `pos_std_m=1.0`, `vel_std_m_s=0.3`, `gyro_std_rad_s=0.0`,
`dropout_prob=0.0`, `estimator_enabled=False`, `estimator_alpha=0.25`,
`v_tol_m_s=0.5`, `h_tol_m=0.05`, `dist_tol_m=1.0`, `attitude_dev_tol_deg=60`,
`omega_tol_deg_s=720`.

## Results (defaults, 100 runs, seed 1234)

Before (velocity-only, `kp_xy=0`) vs after (position + velocity, `kp_xy=0.4`),
fixed target, sensor layer ON (`pos_std=1.0`, no estimator):

| | LANDED | OFF_TARGET | v_end mean / max | dist mean / max |
|---|---|---|---|---|
| `kp_xy = 0` (legacy) | 18 | 82 | 0.405 / 0.470 | 1.77 / 4.50 |
| `kp_xy = 0.4` (position loop) | **100** | 0 | 0.414 / 0.469 | **0.127 / 0.323** |

The random start offset is `0.24..1.50 m` (mean `1.05`); the position loop drives
the final target distance down to a mean of `0.13 m` (max `0.32`), so the mean
offset is removed by ~89%, and the touchdown speed stays well under `v_tol`.

CLI output of the new default series:

```
Seed: 1234  |  Laeufe: 100  |  LANDED: 100  fehlgeschlagen: 0  (100%)
Status: LANDED=100  TOO_FAST=0  OFF_TARGET=0  CONTROL_LOSS=0  TIMEOUT=0  NO_DATA=0
Aufsetzgeschwindigkeit [m/s]: mean=0.41 median=0.41 max=0.47 (n=100)
Zielabstand [m]: mean=0.13 max=0.32 (n=100)
```

### The 100% is honest - it is the *gentle defaults*, not the controller being magic

The default sampled disturbances are deliberately small (tilt <= 6 deg,
`|omega_b| <= 20 deg/s`, `|v_h| <= 0.5 m/s`, start offset <= 1.5 m) and the
descent is slow (`0.4 m/s`) with a long window (`tf = 60 s`), so the position loop
(~2.5 s time constant) has ample time. Under **larger** disturbances the mode does
fail, and those failures are real physical/actuator limits, not tuned away:

| stress config (100 runs, seed 1234, sensor off) | result |
|---|---|
| tilt 30 deg, `omega` 200 deg/s, offset 4 m, `v_h` +-2, `v_v` +-2 | 49 LANDED, 32 TOO_FAST, 18 CONTROL_LOSS - dist mean 10 m, `v_end` max 25 m/s |
| `dist_tol = 0.1 m` | 94 LANDED (position loop accurate to ~0.18 m); 6 OFF_TARGET |
| `dist_tol = 0.02 m` | 85 LANDED |
| commanded `vz_des = -1.0 m/s` (> `v_tol`) | 0 LANDED, 100 TOO_FAST |

The last row is a **structural** limit: the controller holds a constant sink rate
with no terminal flare, so any commanded descent faster than `v_tol` lands
`TOO_FAST` no matter what the horizontal loop does.

The small-gimbal case is *not* a limit: even with `gimbal_max_rad = 0.1` (5.7 deg)
the position loop still lands, because the gimbal torques the whole BODY and the
body tilt (not the nozzle angle) redirects the thrust; the gimbal limit bounds the
attitude rate, not the achievable lean.

The other failure modes stay reachable and are exercised by the tests:
`TOO_FAST` (low start + fast downward speed), `OFF_TARGET` (position loop off),
`CONTROL_LOSS` (tight rate tolerance), `TIMEOUT` (too-short `tf`).

## Known simplifications / limits

- No drag, no aerodynamics, no real engine, no real attitude controller.
- The horizontal POSITION loop is a virtual cascade of two P/PD-like loops
  (position -> velocity -> tilt -> attitude -> gimbal), not a real guidance law.
  With the observation layer it corrects the **estimated** position; the estimate
  RMS is ~1.7 m, yet the terminal offset stays ~0.13 m because the zero-mean
  measurement noise is averaged out over the long descent.
- No attitude sensor: the observed state forwards the true `q`.
- Constant sink rate, no terminal flare -> a commanded descent faster than
  `v_tol` always lands `TOO_FAST` (structural, see above).
- Fuel burn is disabled (`isp_s = 0`), so mass is constant within a run.
- The statistics describe THIS simplified controller and these landing rules;
  they are not predictions about a real rocket and must not be used for real
  autoland design.
