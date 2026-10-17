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

## Virtual TVC servos (`physics/servo.py`)

The two gimbal angles are not applied instantly: two **independent virtual
servos** sit between the command and the thrust model. This is a MODEL, not a
real actuator.

- **Servo X** turns the nozzle about the body `+X` axis (angle `gimbal_x`),
  **Servo Y** about the body `+Y` axis (angle `gimbal_y`). The two axes are
  independent.
- The command (controller or manual slider) is only a **setpoint**. The servo
  produces the **actual** angle through:
  - an angle limit `|angle| <= gimbal_max_rad`,
  - a max slew rate `|d(angle)/dt| <= servo_rate_max_rad_s`,
  - a first-order response with time constant `servo_tau_s`.
- The setpoint is clamped to the physical limit before it is followed, so no
  command can drive the nozzle past `gimbal_max_rad`. The actual angle can never
  jump to the setpoint in one step and stays finite for every `dt > 0`.
- The **actual** angle drives `ThrustModel` -> force `F_b = T * n_b(gx, gy)` and
  moment `M_b = r_b x F_b`; the setpoint never touches the physics directly.

Rotation axes at the joint: the servo angles rotate the thrust direction with
`n_b = R_y(gy) R_x(gx) [0,0,1]` in the BODY frame; the whole vector is then
rotated to the world with `q`, so the nozzle direction correctly inherits the
rocket orientation.

## Integration order (`simulation/flight.py`)

Per step, deterministic and documented:

1. Decide the command setpoints: manual -> `(manual_thrust_N, manual_gimbal_x/y)`
   (hover thrust `m*g` if `manual_thrust_N <= 0`), else the model controller ->
   `(T, gx, gy)`, else open-loop `(thrust_N_open, gimbal_x_open, gimbal_y_open)`.
2. Run the two virtual servos: clamped setpoint -> rate-limited **actual** angle
   `(gx, gy)` (`physics/servo.py`).
3. Build the wrench from the ACTUAL angle:
   `F_b, M_b = thrust.force_moment(T, gx, gy)`.
4. Rotate thrust to inertial: `F_i = rotate(q, F_b)`; gravity
   `F_g = m * [0, 0, -g0]` (inertial, mass dependent).
5. Update translation (Euler-Cromer) with `F = F_i + F_g`. This changes
   `pos/vel/acc` ONLY.
6. Update rotation (semi-implicit) using `M_b` and the Euler equations. This
   changes `q/omega_b/alpha_b` ONLY.
7. Optionally burn fuel if `isp_s > 0` (MODEL), then ground handling.

The same `q` rotates the thrust and feeds the rotation update, so translation and
rotation are genuinely coupled: the orientation determines the thrust direction.

## Model controller (virtual, not a real autopilot)

`_model_control` is a small, documented learning model with three cascaded loops:
- **Descent (vertical)**: `T = m (g + kv_vz (vz_des - vz))`, clamped.
- **Lateral (horizontal) velocity**: turn the horizontal-velocity error into a
  commanded tilt of the thrust axis,
  `tilt = -kv_vh * (v_h - v_h_cmd) / (T/m)`, clamped to `+-max_tilt_deg`. Tilting
  the body is the only mechanism that produces the horizontal thrust component
  that opposes the drift. `kv_vh = 0` disables it and degenerates to the older
  attitude-only model.
- **Attitude hold (inner)**: drive the body `+Z` onto the *desired* thrust
  direction `u_des_world` (vertical, or the tilted axis from the lateral loop).
  In the body frame `u_b = R^T u_des_world`, `e_b = cross(z_b, u_b)`;
  command `alpha_des = -kp e_b - kd omega_b`; convert to a desired body moment
  and then to gimbal angles via `gimbal_for_moment`.

The lateral setpoint `v_h_cmd` comes from `_horizontal_velocity_command`:
- base value is the feed-forward `vh_des_m_s` (velocity damping only), and
- **if** a `target_pos` and `kp_xy > 0` are given, an outer **horizontal
  POSITION** loop adds `kp_xy * (target_xy - pos_xy)`, clamped to
  `vh_max_m_s`. This is a position-to-velocity outer loop around the existing
  velocity loop: the vehicle is steered back over the target instead of only
  being stopped. `kp_xy = 0` (the default) / `target_pos = None` reproduces the
  velocity-only behaviour **exactly** (pinned by a test). The target is a known
  mission constant, so the loop itself needs no sensor; through the observation
  layer the *position* it corrects is the estimated one.

This turns the nozzle into a real actuator, so the displayed gimbal is a
physical input (no independent sine) and the rotation stays bounded.

The position loop is what the **random-scenario mode** uses (target = a fixed
landing pad). With `kp_xy = 0` a start offset / drift can only be damped, never
removed, so those runs end `OFF_TARGET`.

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
`omega_b`, `alpha_b`, dry/fuel/total mass, thrust magnitude, the **actual servo
angles** (`gimbal_x_rad`/`gimbal_y_rad`) kept separate from the **setpoints**
(`gimbal_x_cmd_rad`/`gimbal_y_cmd_rad`), the servo rates and saturation flags,
the manual/auto flag, thrust `F_Schub` (components, magnitude, direction,
application point), gravity
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

### Rocket, nozzle and thrust direction

The hull and the movable nozzle are rebuilt **every frame** from the current
state (position, orientation `q`, gimbal angles) - never a static or fake
overlay. The nozzle is hinged at the base centre and rotated by the **same**
gimbal angles the physics uses, so its axis equals `-thrust_direction_body(gx, gy)`
(pinned by a test). The view draws the body long axis (`+Z`, green,
"Laengsachse") and the **actual thrust direction** (`+n_b` rotated by `q`, orange,
"Schubrichtung") as two separate lines; they coincide only at zero gimbal, and
the angle between them (plus the gimbal angles) is shown as text.

### Camera (interactive, never jumps per frame)

The camera is owned by the view; a per-frame `update()` never reframes it. Only
an explicit reframe (view-mode switch or "Ansicht reset") repositions it.

- rotate: left-drag on the 3D axes
- pan: right/middle-drag
- zoom in/out: mouse wheel, `+`/`-` keys, or the Zoom +/- buttons
- reset: the "Ansicht reset" button (or `r`)
- keys: `o` = overview, `c` = close-up

Built-in matplotlib mouse rotation is disabled so a user rotation is never
silently overwritten by the view.

### Manual servo control (Servo X / Servo Y sliders)

A control-mode switch selects **Auto (Regler)** or **Manuell (Servo)**, and two
labeled sliders set the manual servo setpoints `Servo X`/`Servo Y` in degrees
(spanning `+-gimbal_max_rad`).

- In **Auto** the sliders are inert; the controller provides the setpoints.
- In **Manuell** the sliders provide the two setpoints. Moving a slider
  **recomputes the run from t=0** and the view states this on screen (the status
  line shows `MANUELL ... neu berechnet (t=0)`), so a restart is never silent.
  Manual mode keeps the vehicle aloft with a hover thrust (`m*g`) so the servo
  effect stays visible.
- The status panel always shows the active mode, the servo limits, and the
  setpoint vs. actual angle, so it is clear whether manual or automatic control
  is active.

### Servo axes on the 3D model

At the engine joint (hinge) the view draws the two **servo rotation axes**
(Servo X about body +X, red; Servo Y about body +Y, blue), rotated by the rocket
orientation, plus the nozzle axis ("Duesenachse") and the actual thrust direction
("Schubrichtung"), so both servo axes and the current nozzle tilt are visible.

### View modes

- **Gesamtansicht** (overview): whole trajectory, ground plane and force arrows.
- **Nahansicht** (close-up): the camera tracks the moving rocket so the hull and
  the nozzle tilt are clearly visible; the follow only moves the centre and keeps
  the user's rotation and zoom.

### Force arrows

Velocity/acceleration/force arrows are scaled by their **own** physical magnitude
(normalized to the running maximum of the same quantity) and capped at a fraction
of the current view size (`0.35 * half-size`) - independent of the rocket
geometry - so they stay visible and never swamp the hull.

### Single-animation rule (macOS)

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
- `tests/test_servo.py` - angle-limit clamping, slew-rate limiting, first-order
  approach without overshoot, reported rate equals the applied change, safe for
  `dt <= 0`, two independent axes, sign handling, parameter validation.
- `tests/test_flight.py` (servo additions) - setpoint vs. actual separation,
  actual angle never exceeds the limit, the controller respects the limits, the
  servo rate is bounded by config, manual servo X/Y move the thrust in the
  correct body direction and produce the matching moment, manual mode overrides
  the controller and keeps hover thrust, force AND moment coupling, telemetry
  exposes setpoint/actual/rate/saturation, and the full suite still holds.
- `tests/test_flight_view.py` - multi-step 3D update, labeled vectors, no
  independent-sine nozzle, hull/nozzle rebuilt from the actual state each frame,
  nozzle axis tied to the physics thrust direction, thrust-vs-long-axis differ
  when gimballed, force arrows scaled independently of the rocket geometry,
  same-state-source telemetry, HUD lateral speed matches the state (no
  display/frame error), interactive camera (rotate/zoom/pan) that does NOT jump
  per frame, overview/close-up modes, restart / dt / mode, and restart reuses a
  single animation (regression for the macOS timer crash). Plus the TVC servo
  controls: the two sliders and the control-mode switch exist with the correct
  range, both servo axes are drawn, manual mode recomputes and is clearly marked
  (no silent reset), the sliders are inert in auto mode, and the telemetry
  separates setpoint from actual.

## Not modelled (explicit)

Aerodynamics / drag, engine dynamics and real thrust control, structural
flexibility, real sensors/estimation (see the sensor experiments), and any real
vehicle behaviour. The TVC servos are a deliberately simple first-order +
rate-limited model - no real servo motor, gearbox, backlash, current limit or
thermal behaviour is represented. This is a learning visualization only.
