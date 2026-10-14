# selflanding_rocket

Modular 3D rocket simulation focused on:
- Physically traceable dynamics (translation + rotation)
- Sensor models with uncertainty (noise, bias, drift, delay, dropout)
- State estimation with uncertainty awareness
- Transparent system monitoring & failure logging
- Reproducibility, validation, and systematic fault injection

## Quickstart

Run console simulation (generates CSV + plots in `output/`):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 main.py
```

Run 3D visual simulation (interactive, uses real sim data):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 visual_app.py
```

Run tests:

```bash
cd selflanding_rocket
PYTHONPATH=. python3 -m pytest -v
```

## 3D Visual (matplotlib)

- 3D scene: ground plane, axes (X=red, Y=blue, Z=green), rocket marker, flight trail
- Uses actual State history from Python simulation (no made-up trajectory)
- Controls: Pause/Play, Restart
- HUD: t [s], h [m], v [m/s]
- Notes: Rotation stored but not integrated in current dynamics; trajectory = true translation. No autoland.

## Output files (main.py)

- `output/trajectory.csv` – true simulation state
- `output/summary.json` – basic summary
- `output/height.png`, `output/velocity.png`

View plots: `open output/height.png` (macOS) or open in Finder.

## Architecture
(see existing structure; visualization now includes `viz3d_mpl.py` + `visual_app.py`)

## Landing 3D Visual (Lernversion)

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_app.py
```

Features: Zielpunkt (X,Y) per Slider änderbar, Start/Pause/Neustart, Anzeige t/h/v/dist(Ziel). Nutzt echte Simulationsdaten. Steuerung/Landung nicht implementiert (klar gekennzeichnet).

## Autoland Learning Studio (interactive, full UI)

The main interactive environment. Configure the whole scenario, run / pause /
reset / recompute, and see the 3D path plus height / speed / distance plots at
the same time. Simplified point-mass model only - NOT a realistic rocket.

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_studio_app.py
```

### Step-by-step operation
1. Bottom text boxes = scenario settings: start height [m], horizontal start
   offset X/Y [m], target X/Y [m], initial vertical velocity `v0z` [m/s],
   initial horizontal velocity `v0x` [m/s] (along +X), time step `dt` [s].
2. Type a value and press **Enter** in a box to accept it.
3. Press **Neu berechnen** (or **Start**) to run a FRESH simulation with the
   current settings. Every run is recomputed - never a replayed old path.
4. **Pause / Weiter** pauses and resumes the animation.
5. **Reset** rewinds the current run to t=0 (no recompute).
6. Left: 3D path (black), rocket (red dot), target (blue x), ground plane.
7. Right: height, speed and target-distance curves over time; the dot marks
   the current frame.
8. On the 3D view: the top-left block is labelled `AKTUELL` (current frame:
   t, h, v_z, v_xy, dist, control ON/OFF phase). The top-right block is
   labelled `GESAMTERGEBNIS (berechnet)` and shows the final landing result
   of the whole run. This makes clear which values are current and which
   belong to the completed run.
9. Results: `LANDED` (soft, on target), `TOO_FAST` (touchdown too fast),
   `FAILED` (no valid touchdown / timeout / tunnelled through the ground).

Code separation: scenario/runner `simulation/scenario.py`; presentation
`visualization/viz3d_studio.py`; physics `simulation/loop.py`; control
`physics/controller.py`; assessment `evaluation/landing.py`.

## Experiment / evaluation mode

Runs a bounded set of parameter sweeps automatically (start height, target
distance, initial velocities, time step, and a timeout check), prints a result
table and writes a CSV plus a summary plot. It reuses the SAME simulator,
controller and evaluator as the studio - no second physics implementation.

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_experiment_app.py          # table + CSV + PNG
PYTHONPATH=. python3 landing_experiment_app.py --show   # also opens the plot
```

Output:
- `output/experiment/experiment_results.csv` - one row per trial (all settings
  + all measurements)
- `output/experiment/experiment_results.png` - 5 subplots (one per sweep)

### Sweeps (documented, bounded)
| sweep | values | meaning |
|---|---|---|
| `start_height_m` | 5, 10, 20, 40, 60 | start height [m] |
| `target_dist_m` | 0, 1, 2, 4, 8, 12 | target at (d, 0), horizontal distance [m] |
| `v0_vertical_m_s` | 0, -1, -2, -5, -10 | initial vertical velocity [m/s] |
| `dt` | 0.1, 0.05, 0.02, 0.01, 0.005 | time step [s] |
| `max_time_s` | 2, 5, 20, 60, 300 | run cut off after this time [s] |

### How to read the table
```
sweep             value    status   dur[s]   h_end  vz_end  vxy_end   dist   steps
max_time_s        2.000    FAILED     1.95   24.29   -0.48     0.00   0.00      40
max_time_s       60.000    LANDED    50.60    0.00   -0.50     0.00   0.00    1013
```
- `status`: `LANDED` (soft, on target), `TOO_FAST` (touchdown > 0.5 m/s),
  `FAILED` (no valid touchdown; includes timeout while airborne).
- `dur[s]`, `h_end`, `vz_end`, `vxy_end`, `dist`, `steps`: measured at the
  final simulated state. For a timeout these describe the airborne state at
  cut-off - and it is still `FAILED`, never `LANDED`.

### How to read the plots
Per sweep: black line = total touchdown speed, red dashed line = 0.5 m/s
limit, dotted blue = simulation duration (right axis). Green circle = `LANDED`,
red cross = everything else. `dt` uses a log x-axis.

### Open the CSV
- Numbers: double-click the file.
- Excel: "Data -> From Text/CSV", comma separator.
- Python:
  ```python
  import pandas as pd
  df = pd.read_csv("output/experiment/experiment_results.csv")
  print(df.groupby("sweep")["success"].mean())
  ```

### What these results do NOT tell you
The model is a simplified point mass: no thrust limits, no attitude/rotation,
no engine dynamics, no drag, no sensors/estimation, horizontal start velocity
only along +X. So the numbers show how THIS simplified guidance + landing
rules behave - they are **not** predictions about a real rocket and must not be
used for real autoland design.

## Sensor uncertainty & state estimation (learning experiment)

Compares four groups under IDENTICAL initial conditions, so the only thing that
changes is what the controller is allowed to see:

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_sensor_experiment_app.py            # statistical run
PYTHONPATH=. python3 landing_sensor_experiment_app.py --show     # also opens the plot
PYTHONPATH=. python3 landing_sensor_experiment_app.py --seeds 1234,7,42
```

| group | controller sees | estimator |
|---|---|---|
| `ideal` | the TRUE state | none (reference) |
| `noise` | noisy measurements only | none |
| `dropout` | noisy + occasionally missing samples | none |
| `estimation` | noisy + missing samples | simplified low-pass filter |

Strict separation (this is the point of the experiment):
- A **measurement** never modifies the true state; the true state is only read
  (a copy is taken) to synthesise a sensor value.
- The controller receives the **estimate**, never the truth. In `ideal` the
  estimate equals the truth; in the other groups it does not.

Documented, purely virtual sensor parameters
(`SensorConfig`, defaults): position noise std `1.0 m`, velocity noise std
`0.3 m/s`, no bias, dropout probability `0.1`. Estimator:
`LowPassEstimator(alpha=0.25)` (`x <- x + alpha*(z - x)`); a missing sample is
skipped and the previous estimate is held. All randomness comes from one seeded
`numpy.random.default_rng`, so a run is reproducible.

### Reproducible statistical run (5 seeds, 100 trials)
The whole comparison is repeated once per seed. Every pass uses the SAME
scenarios and initial conditions and the SAME controller parameters; only the
sensor RNG seed changes. One seed is fully deterministic. Defaults: 5 scenarios
(`h20_d0`, `h25_d3`, `h40_d6`, `h30_d2y`, `h60_d10`) x 4 groups x 5 seeds
(`1234, 101, 202, 303, 404`) = **100 trials**. Controller parameters and
physics are never changed between modes.

Output:
- `output/sensor_experiment/sensor_experiment_trials.csv` - one row per trial
  incl. its `seed`: group, scenario, status, success, contact, end speed,
  target distance (at contact), estimate RMS error, missing samples, steps,
  duration.
- `output/sensor_experiment/sensor_experiment_summary.csv` - one row per
  (group, scenario) plus one `ALL` row per group: trial count, success /
  per-status counts (`LANDED`, `TOO_FAST`, `FAILED`, `NO_DATA`), success rate,
  mean & max end speed, mean target distance at contact, estimate RMS error,
  seeds used.
- `output/sensor_experiment/sensor_experiment_statistics.png` - success rate
  per group with one point per seed, stacked status counts, mean/max end speed,
  and estimate RMS error.

Example result (defaults, 5 seeds):
```
mode        trials  LND  FAST  FAIL  NOD   succ   v_mean  v_max   dist*   e_pos
ideal          25    25    0     0    0   100%    0.500  0.500    0.00   n/a
noise          25    25    0     0    0   100%    0.351  0.461    0.18   1.737
dropout        25    25    0     0    0   100%    0.336  0.444    0.21   1.794
estimation     25    17    8     0    0    68%    0.471  0.588    0.17   0.802
```
*`dist*` = mean target distance at ground contact; `LND/FAST/FAIL/NOD` =
per-status counts. Failed or aborted runs stay in the table (`FAIL`/`NOD`
columns), so no trial silently vanishes from the statistics.*

Stability across seeds: `ideal`, `noise` and `dropout` land 25/25 every time.
`estimation` is the only failing mode (17/25, always `TOO_FAST`); the previous
single-run finding is therefore stable, not a fluke: the low-pass filter
roughly halves the estimate error (pos 1.74 -> 0.80 m) but LAGS - the measured
descent speed underestimates the true one, the controller brakes a little late
and touches down just over the 0.5 m/s limit. Filtering trades noise for lag;
it is not automatically "better". No filter was tuned - this only reports how
stable the current results are, before any optimisation.

### Diagnostics: when does the estimate lag show up?

Explains the `TOO_FAST` cases by plotting, per failed run, the TRUE velocity,
the raw (noisy) measurements and the ESTIMATE over time, and by quantifying the
deviation.

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_sensor_diagnostics_app.py          # default: 5 seeds
PYTHONPATH=. python3 landing_sensor_diagnostics_app.py --show
PYTHONPATH=. python3 landing_sensor_diagnostics_app.py --seeds 1234,202
```

Uses the SAME scenarios and seeds as the experiment (re-runs the exact failing
(scenario, seed), so results are identical). The time series is recorded live
during the real simulation run by `sensors/observed.py` (`ObservedController.
record`): per step it stores TRUE state, raw MEASUREMENT (incl. dropouts) and
ESTIMATE. The figures therefore show real stored simulation data - no synthetic
example curves (verified by a test that compares the plotted line data to the
recorded series).

Per failure a 3-panel figure
`output/sensor_diagnostics/failures/estimation__<scenario>__seed<s>.png`:
1. height h(t); 2. vertical velocity vz (true line, measured dots with visible
   dropout gaps, estimate line); 3. speed |v| vs the 0.5 m/s limit. Ground
   contact is marked on all panels and the run is labelled `TOO_FAST` with its
   touchdown speed. A summary `output/sensor_diagnostics/failures.csv` lists per
   failure: touchdown speed, true/estimated vz at touchdown (their difference =
   lag), max |deviation| and its time, mean |deviation| over the last 2 s, etc.

Example (default run, 8 failures out of 25 estimation trials):
```
scenario    seed   v_end  vz_true   vz_est   delta max|dev|  t_max |dev| letzte 2s
h30_d2y     1234   0.522   -0.496   -0.173   0.323    0.874   0.00           0.075
h60_d10     1234   0.564   -0.566   -0.518   0.048    0.874   0.00           0.087
h25_d3       101   0.509   -0.480   -0.211   0.269    0.475  25.50           0.103
...
```
Reading: at touchdown the estimate is usually NOT settled on the true value
(Δ vz `0.04..0.32 m/s`), i.e. the filter is still lagging the final transient,
so braking comes a moment too late. The largest deviation appears at t=0 (the
filter starts from zero -> first estimate jumps) or when descent begins
(t~25-65 s for high drops). Over the last 2 s the estimate wanders around the
truth by `~0.08-0.14 m/s` (noise + lag), which is enough to push some landings
just over 0.5 m/s. Diagnosis only - no filter was optimised.

### What this does NOT model
Simplified point mass, scalar white Gaussian noise + constant bias, independent
per-channel dropouts, no delay, no IMU integration, no Kalman filter, no sensor
validation. It shows how THIS simplified guidance reacts to noise and lag - it
is **not** a realistic sensor/estimation model and says nothing about real
hardware.

## Autoland (Learning Simulation, controller coupled)

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_ctrl_app.py
```

Shows: actual simulated path, target point, distance, height, speed, whether
guidance is ON/OFF and the live landing status (`IN_FLIGHT`, `LANDED`,
`TOO_FAST`, `FAILED`). Moving the target sliders re-runs the simulation live
(the scene is reloaded), it does not merely replay a stale trajectory.

Run the automated multi-scenario report (end status/height/speed/distance):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 evaluation/landing_matrix.py
```

Controller contract (`physics/controller.py`):
`compute(state, target_pos, mass_kg) -> a_cmd` returns a thrust acceleration
(m/s^2) WITHOUT gravity; the simulator adds gravity separately, so
`a_total = g + a_cmd` (no double gravity). `Simulator(..., guidance=None)`
keeps the old gravity-only behaviour unchanged.

Landing evaluation (`evaluation/landing.py`): `LANDED` requires ground contact
within tolerance AND total speed `<= 0.5 m/s` AND horizontal distance to the
target `<= 1.0 m`. See `docs/autoland_design.md`.
