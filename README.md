# selflanding_rocket

Modular 3D rocket simulation focused on:
- Physically traceable dynamics (translation + rotation)
- Sensor models with uncertainty (noise, bias, drift, delay, dropout)
- State estimation with uncertainty awareness
- Transparent system monitoring & failure logging
- Reproducibility, validation, and systematic fault injection

## Quickstart

```bash
pip install -e .
python main.py
```

Run tests:

```bash
pytest -v
```

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
- `visualization/` – 3D + telemetry plots
- `evaluation/` – metrics & reports
- `tests/` – unit/integration tests
- `docs/` – model docs, assumptions
- `data/` – reference/test data

## Conventions

- Units: SI (meters, kg, s, rad, N, Pa). See `config/units.py`.
- Coordinate frame: ENU-like local vertical frame for simplicity (X East, Y North, Z Up). Documented in `docs/conventions.md`.
- Time: fixed or configurable dt; reproducible via seeds.
- Separation: true state, raw sensor readings, filtered estimates are kept distinct.
