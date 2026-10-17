"""Virtual TVC servo model: two INDEPENDENT single-axis servos (learning only).

This is a deliberately simple, tunable MODEL of the two servos that move the
gimbaled nozzle. It is NOT a real servo/actuator model. Each servo turns the
nozzle about ONE body axis:

    Servo X -> rotation about the body +X axis (pitch-like), angle `gimbal_x`
    Servo Y -> rotation about the body +Y axis (yaw-like),  angle `gimbal_y`

The two axes are independent (no coupling between them). The servo turns the
COMMANDED setpoint into an ACTUAL angle through:

    * an angle limit  |angle| <= angle_max_rad  (configurable),
    * a slew-rate limit |d(angle)/dt| <= rate_max_rad_s (configurable),
    * a first-order response with time constant `tau_s` (configurable).

Because the response is rate-limited and first order, the actual angle can never
jump to the setpoint in one step: it moves smoothly and stays finite for every
dt > 0. `tau_s = 0` means "as fast as the rate limit allows".

Contract (documented + tested):
- `step(setpoint, dt)` returns the new ACTUAL angle and never exceeds the angle
  limit, regardless of how large the setpoint or dt is.
- The reported `rate` is the actual applied rate `(angle_new - angle_old)/dt`.
- Signs: a positive setpoint moves a positive angle; the mapping to thrust
  direction lives in `physics/thrust_model.py`.
"""
import numpy as np


class Servo:
    """One virtual single-axis servo (angle limit + slew rate + first order)."""

    def __init__(self, angle_max_rad: float = 0.35,
                 rate_max_rad_s: float = 2.0,
                 tau_s: float = 0.05,
                 angle0_rad: float = 0.0):
        if angle_max_rad < 0.0:
            raise ValueError("angle_max_rad must be >= 0")
        if rate_max_rad_s < 0.0:
            raise ValueError("rate_max_rad_s must be >= 0")
        if tau_s < 0.0:
            raise ValueError("tau_s must be >= 0")
        self.angle_max_rad = float(angle_max_rad)
        self.rate_max_rad_s = float(rate_max_rad_s)
        self.tau_s = float(tau_s)
        # actual angle and the setpoint it is driving towards
        self.angle = float(np.clip(angle0_rad, -self.angle_max_rad,
                                   self.angle_max_rad))
        self.setpoint = self.angle
        self.setpoint_request = self.angle
        self.rate = 0.0

    def clamp_setpoint(self, setpoint: float) -> float:
        """Clamp a commanded setpoint into the physical angle limits."""
        return float(np.clip(setpoint, -self.angle_max_rad, self.angle_max_rad))

    def step(self, setpoint: float, dt: float) -> float:
        """Advance the servo by `dt` towards `setpoint`; return the actual angle.

        The setpoint is clamped to the angle limit first (the servo can never be
        asked to exceed its physical range), then the motion is limited by the
        first-order time constant AND the slew-rate limit.
        """
        if dt <= 0.0:
            return self.angle
        self.setpoint_request = float(setpoint)
        sp = self.clamp_setpoint(setpoint)
        self.setpoint = sp
        err = sp - self.angle
        if self.tau_s > 0.0:
            rate_des = err / self.tau_s
        else:
            rate_des = err / dt
        rate = float(np.clip(rate_des, -self.rate_max_rad_s, self.rate_max_rad_s))
        new_angle = self.angle + rate * dt
        # never overshoot the (clamped) setpoint in a single step
        if (sp - self.angle) * (sp - new_angle) < 0.0:
            new_angle = sp
        new_angle = float(np.clip(new_angle, -self.angle_max_rad,
                                  self.angle_max_rad))
        self.rate = (new_angle - self.angle) / dt
        self.angle = new_angle
        return self.angle

    def reset(self, angle0_rad: float = 0.0) -> None:
        self.angle = float(np.clip(angle0_rad, -self.angle_max_rad,
                                   self.angle_max_rad))
        self.setpoint = self.angle
        self.setpoint_request = self.angle
        self.rate = 0.0

    @property
    def saturated(self) -> bool:
        """True if the REQUESTED setpoint is outside the physical angle limit."""
        return abs(self.setpoint_request) > self.angle_max_rad + 1e-12


class ServoModel:
    """The two independent TVC servos (X about body +X, Y about body +Y)."""

    def __init__(self, angle_max_rad: float = 0.35,
                 rate_max_rad_s: float = 2.0,
                 tau_s: float = 0.05,
                 angle_x0_rad: float = 0.0,
                 angle_y0_rad: float = 0.0):
        self.angle_max_rad = float(angle_max_rad)
        self.x = Servo(angle_max_rad, rate_max_rad_s, tau_s, angle_x0_rad)
        self.y = Servo(angle_max_rad, rate_max_rad_s, tau_s, angle_y0_rad)

    def step(self, setpoint_x: float, setpoint_y: float, dt: float):
        """Advance both servos; return the two ACTUAL angles (x, y)."""
        ax = self.x.step(setpoint_x, dt)
        ay = self.y.step(setpoint_y, dt)
        return ax, ay

    def reset(self, angle_x0_rad: float = 0.0, angle_y0_rad: float = 0.0) -> None:
        self.x.reset(angle_x0_rad)
        self.y.reset(angle_y0_rad)

    @property
    def angle_x(self) -> float:
        return self.x.angle

    @property
    def angle_y(self) -> float:
        return self.y.angle

    @property
    def setpoint_x(self) -> float:
        return self.x.setpoint

    @property
    def setpoint_y(self) -> float:
        return self.y.setpoint

    @property
    def rate_x(self) -> float:
        return self.x.rate

    @property
    def rate_y(self) -> float:
        return self.y.rate
