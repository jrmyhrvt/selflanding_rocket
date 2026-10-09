# Physics Model

## Coordinate System
- Local vertical frame (ENU-like): X East, Y North, Z Up.
- Flat-Earth approximation acceptable for initial development.

## Translation
- Numerical integration: semi-implicit Euler (Euler-Cromer) – `vel += a*dt`, `pos += vel*dt`.
- Forces summed in inertial frame: `a = F_total / m`.
- Gravity: `g = [0,0,-g0]`, `g0 = 9.80665 m/s^2`.

## Rotation
- Quaternions scalar-first [w,x,y,z]. Normalized after updates.
- Euler angles for visualization only.
- Full rotational dynamics (Euler's equations) to be connected once inertia/moments defined.

## Atmosphere
- ISA troposphere model (linear lapse). Limited validity; behavior above troposphere not modeled.
- Air density `rho(h)` decreases with height; zero/very low air density at high altitude not physically correct beyond model scope.

## Mass
- `m_total = m_dry + m_fuel`. Fuel burned separately; mass change affects acceleration.
- CG/inertia tracked separately when used.

## Simplifications & Known Limits
- No drag/wind/thrust yet (loop applies only gravity).
- No Coriolis/centrifugal (flat Earth).
- No 6DOF coupling active yet.
- Numerical errors scale with dt; dt=0.01 s reasonable for basic cases.
