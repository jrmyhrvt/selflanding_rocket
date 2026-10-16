# Unified 3D flight simulation (learning model)

This document specifies the joint translation + rotation simulation added to
make the 3D flight physically traceable. It is a **learning model**, not a
real rocket, engine or autopilot.

## Why (root cause of the old "uncontrolled rotation")

Before this change rotation was never part of the flight. The rotation demo set
a **constant body-frame moment** once on the state and never changed it:

- `simulation/scenario.py::run_rotation_demo` sets `moment_total = [2,0,0]`.
- `physics/rotation.py::integrate_rotation_body` reads `state.moment_total`
  every step.
- `physics/rigidbody.py::angular_accel` gives
  `alpha_b = I^{-1} (M_b - omega_b x (I_b omega_b))`.
- With a constant `M_b` and no damping the update `omega += alpha*dt` makes
  `omega` grow **linearly** -> a steadily accelerating tumble (spin-up). That is
  the "uncontrolled rotation".

It was **not** a bug in the time step, units, quaternion normalization or the
integration order:
- `q` is re-normalized after every step (`physics/rotation.py`),
- the semi-implicit order (`omega` updated, then `q`) is consistent,
- SI units and the ENU frame are used throughout.

The real gaps were: (1) no physical actuator produced the moment, and (2) no
force/moment coupling between translation and rotation. The unified model adds a
bounded gimbaled thrust that produces BOTH force and moment.

## Frames and conventions

- Inertial frame: ENU, `x` East, `y` North, `z` Up. World vectors (thrust,
  gravity, velocity, acceleration) are inertial.
- Body frame: `+Z` along the rocket long axis (base at `z=0`, tip at `+length`).
- `q = [w, x, y, z]` scalar-first is `q_{B->I}` (rotates body -> inertial).
- `omega_b`, `alpha_b`, `M_b` are body-frame.
- Nozzle hinge sits at `r_b = [0, 0, -offset]` below the base.

## Simplified thrust model (`physics/thrust_model.py`)

For a commanded thrust magnitude `T` and gimbal angles `(gx, gy)`
(`gx` about body X, `gy` about body Y):

```
n_b = [cos(gx) sin(gy), -sin(gx), cos(gx) cos(gy)]   # unit thrust dir, body
F_b = T * n_b
M_b = r_b x F_b = [offset*F_b_y, -offset*F_b_x, 0]   # no roll authority
```

- `T` is clamped to `[0, max_thrust_N]`, angles to `+-gimbal_max_rad`.
- `gimbal_for_moment(M_b, T)` inverts the relation (best effort, clamped).
- This is a MODEL: no engine dynamics, no propellant flow physics, no plume.

## Integration order (`simulation/flight.py`)

Per step, deterministic and documented:

1. Decide actuator input: model controller -> `(T, gx, gy)`, or open-loop
   `(thrust_N_open, gimbal_x_open, gimbal_y_open)`.
2. Build the wrench: `F_b, M_b = thrust.force_moment(T, gx, gy)`.
3. Rotate thrust to inertial: `F_i = rotate(q, F_b)`; gravity
   `F_g = m * [0, 0, -g0]` (inertial, mass dependent).
4. Update translation (Euler-Cromer) with `F = F_i + F_g`. This changes
   `pos/vel/acc` ONLY.
5. Update rotation (semi-implicit) using `M_b` and the Euler equations. This
   changes `q/omega_b/alpha_b` ONLY.
6. Optionally burn fuel if `isp_s > 0` (MODEL), then ground handling.

The same `q` rotates the thrust and feeds the rotation update, so translation and
rotation are genuinely coupled: the orientation determines the thrust direction.

## Model controller (virtual, not a real autopilot)

`_model_control` is a small, documented learning model with three cascaded loops:
- **Descent (vertical)**: `T = m (g + kv_vz (vz_des - vz))`, clamped.
- **Lateral (horizontal) velocity**: turn the horizontal-velocity error into a
  commanded tilt of the thrust axis,
  `tilt = -kv_vh * (v_h - v_h_des) / (T/m)`, clamped to `+-max_tilt_deg`. Tilting
  the body is the only mechanism that produces the horizontal thrust component
  that opposes the drift. `kv_vh = 0` disables it and degenerates to the older
  attitude-only model.
- **Attitude hold (inner)**: drive the body `+Z` onto the *desired* thrust
  direction `u_des_world` (vertical, or the tilted axis from the lateral loop).
  In the body frame `u_b = R^T u_des_world`, `e_b = cross(z_b, u_b)`;
  command `alpha_des = -kp e_b - kd omega_b`; convert to a desired body moment
  and then to gimbal angles via `gimbal_for_moment`.

This turns the nozzle into a real actuator, so the displayed gimbal is a
physical input (no independent sine) and the rotation stays bounded.

### Root cause of "the lateral motion is not removed" (fixed)

The original controller had **no horizontal-velocity feedback**: it only held
attitude (world `+Z`) and vertical speed. An existing lateral velocity therefore
was never actively removed — it only changed through the transient axial-thrust
component while the rocket self-righted (and could even grow, e.g. a 15° lean
with `vx = +5 m/s` ended at ~`+7 m/s`, 75 m drift). This was a **controller**
gap, not a display or physics fault (see below). The lateral loop above was
added to close it: for the same case the horizontal speed now ends at ~0 and the
drift drops to ~13 m (residual integral while decelerating).

## Telemetry (`simulation/flight.py::flight_telemetry`)

Read directly from ONE `State`, so the view can never show a different value
than the physics. Includes time, position/height, velocity/acceleration
(including `speed_horizontal_m_s = |(vx, vy)|`), quaternion + roll/pitch/yaw,
`omega_b`, `alpha_b`, dry/fuel/total mass, thrust magnitude, gimbal angles,
thrust `F_Schub` (components, magnitude, direction, application point), gravity
`F_Schwerkraft`, resultant `F_gesamt`, the moment, and explicit
`drag_modeled=False` / `aerodynamics_modeled=False` flags plus a thrust-model
note.

**Coordinate-system check (no display bug):** the HUD shows the *inertial ENU*
velocity `vx, vy, vz` and position `x, y, z`, taken verbatim from `state.vel` /
`state.pos`; the velocity plot's `|v|`, `v_z` and the new `|v_h|` trace come from
the same arrays. Tracing `state.vel -> flight_telemetry -> HUD/plot` matches
exactly (pinned by a test), so the reported lateral-speed symptom was **not** a
display or frame error.

## View (`visualization/flight_view.py`, `flight_sim_app.py`)

One figure: 3D rocket + nozzle, world/body axes, labeled force/velocity/
acceleration vectors, live telemetry panel, height / velocity / rotation plots,
and Start / Pause / Weiter / Neustart plus a `dt` box.

The view owns exactly ONE `FuncAnimation`, created once and reused for the whole
session; Start/Pause toggle its event source and Neustart/dt/mode only reset the
frame counter (they never recreate the animation). This matters on macOS: the
backend's animation timer is a C-level CFRunLoop timer, and dropping/recreating
the animation (e.g. on every "Geneigt"/"Neustart"/dt change) freed a
still-registered native timer and crashed the app with `'TimerMac' object has no
attribute 'callbacks'`. Reusing a single animation avoids the use-after-free.

## Tests

- `tests/test_thrust_model.py` - direction, clamping, `M = r x F`, invertibility,
  and the gimbal-sense check (commanded moment -> gimbal -> correct lateral
  force).
- `tests/test_flight.py` - coupling, quaternion normalization, force-only vs
  moment-only effects, moment torques in the commanded direction, zero moment =>
  no artificial rotation, bounded control, tilted flight stays finite over many
  steps (regression), tilted start + horizontal velocity is actively damped
  (regression) and sign-independent, telemetry consistency and not-modelled
  labels, optional fuel burn.
- `tests/test_flight_view.py` - multi-step 3D update, labeled vectors, no
  independent-sine nozzle, same-state-source telemetry, HUD lateral speed matches
  the state (no display/frame error), restart / dt / mode, and restart reuses a
  single animation (regression for the macOS timer crash).

## Not modelled (explicit)

Aerodynamics / drag, engine dynamics and real thrust control, structural
flexibility, real sensors/estimation (see the sensor experiments), and any real
vehicle behaviour. This is a learning visualization only.
