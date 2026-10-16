# Rotation Integration (Phase 1)

## Changes
- `physics/rotation.py`: added `integrate_rotation(state, dt)`. Assumes `omega` in inertial frame [rad/s] (simplified for learning model), scalar-first quaternion `[w,x,y,z]` (body->inertial). Uses Euler step for `q_dot = 0.5*q ⊗ (0, omega)`, normalizes q after step, updates `omega += alpha*dt` (alpha treated as angular acceleration in same frame).
- `simulation/loop.py`: `SimConfig` gains `integrate_rotation: bool = False` (opt-in). Simulator calls `integrate_rotation` after translation when enabled. Default remains False; existing behavior unchanged.

## Tests
- `tests/test_rotation.py`: 5 tests (helpers, no change when omega zero, simple rotation, disabled by default, enabled changes q; norms preserved).

## Constraints
- `omega` frame assumption is simplified (full rigid body distinguishes body/inertial); acceptable as opt-in learning step.
- Euler integration; small dt recommended. No moment->alpha coupling beyond what caller sets.
- Existing 124 tests still pass; total 129 passed.
