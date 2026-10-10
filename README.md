# selflanding_rocket

Modular 3D rocket simulation focused on:
- Physically traceable dynamics (translation + rotation)
- Sensor models with uncertainty (noise, bias, drift, delay, dropout)
- State estimation with uncertainty awareness
- Transparent system monitoring & failure logging
- Reproducibility, validation, and systematic fault injection

## Quickstart

Run simulation (generates CSV + plots in `output/`):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 main.py
```

Or use the analysis entrypoint:

```bash
cd selflanding_rocket
PYTHONPATH=. python3 run_analysis.py --tf 2.0 --dt 0.01
```

Run tests:

```bash
cd selflanding_rocket
PYTHONPATH=. python3 -m pytest -v
```

## Output files

After running `main.py`, files are written to `output/`:
- `trajectory.csv` – true simulation state over time (t, pos/vel/acc, q, omega/alpha, mass/fuel, forces/moments)
- `summary.json` – basic summary (t0/tf/dt, positions/velocities/masses/heights)
- `height.png`, `velocity.png` – time series plots

## View plots

The PNG files are saved locally. On macOS you can open them with:
```bash
open output/height.png
open output/velocity.png
```
Or double-click the files in Finder.

## Architecture

- `config/` – units, conventions, YAML configs
- `state/` – canonical simulation state
- `physics/` – forces, motion, rotation (quaternions)
- `atmosphere/` – atmosphere & wind
- `mass_model/` – mass, fuel, inertia, CG
- `sensors/` – virtual sensors
- `estimator/` – state estimation
- `system_logic/` – monitoring, plausibility, modes
- `simulation/` – time loop & orchestration
- `visualization/` – 2D time plots (+ animation notes)
- `evaluation/` – CSV export, metrics & reports
- `tests/` – unit/integration tests
- `docs/` – model docs, assumptions
- `data/` – reference/test data
- `output/` – generated results (CSV/JSON/PNG)

## Conventions

- Units: SI (meters, kg, s, rad, N, Pa). See `config/units.py`.
- Coordinate frame: ENU-like (X East, Y North, Z Up). See `docs/conventions.md`.
- Time: fixed dt; reproducible via seeds.
- Separation: true state, raw sensor readings, filtered estimates kept distinct.

## 3D Visual Simulation (matplotlib)

Start the interactive 3D view (uses real simulation data):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 visual_app.py
```

Features:
- 3D scene: ground plane, coordinate axes (X red, Y blue, Z green), rocket marker, flight trail
- Uses actual State history from simulation (no fabricated trajectory)
- Controls: Pause/Play, Restart
- HUD: simulation time t [s], height h [m], speed v [m/s]

Notes: Rotation is stored in state but not actively integrated in current dynamics loop – shown as orientation placeholder not animated; trajectory is true translation from physics. No control/autoland logic is simulated.
