from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class State:
    """Canonical true simulation state."""
    t: float = 0.0
    # translation
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x,y,z]
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [vx,vy,vz]
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # rotation (quaternion scalar-first [w,x,y,z])
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))  # angular vel
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))  # angular acc
    # body-frame rotation (separate, backward compatible)
    omega_b: np.ndarray = field(default_factory=lambda: np.zeros(3))  # angular vel in body frame
    alpha_b: np.ndarray = field(default_factory=lambda: np.zeros(3))  # angular acc in body frame
    # mass
    mass_kg: float = 0.0
    fuel_kg: float = 0.0
    dry_mass_kg: float = 0.0
    # derived
    height_m: float = 0.0
    # forces/moments (optional)
    force_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Thrust/nozzle telemetry (only filled by the unified flight model).
    # FORCE direction: inertial frame. MOMENT direction: body frame.
    thrust_N: float = 0.0
    # gimbal_x / gimbal_y = ACTUAL servo angles (drive the physics).
    gimbal_x: float = 0.0
    gimbal_y: float = 0.0
    # servo setpoints (controller/manual command) and actual servo rates.
    # Setpoint != actual while the virtual servos are slewing/responding.
    gimbal_x_cmd: float = 0.0
    gimbal_y_cmd: float = 0.0
    servo_rate_x: float = 0.0
    servo_rate_y: float = 0.0
    servo_sat_x: bool = False
    servo_sat_y: bool = False
    force_thrust: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_gravity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    thrust_dir_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    modeled_thrust: bool = False

    def copy(self) -> "State":
        return State(
            t=self.t,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            acc=self.acc.copy(),
            q=self.q.copy(),
            omega=self.omega.copy(),
            alpha=self.alpha.copy(),
            omega_b=self.omega_b.copy(),
            alpha_b=self.alpha_b.copy(),
            mass_kg=self.mass_kg,
            fuel_kg=self.fuel_kg,
            dry_mass_kg=self.dry_mass_kg,
            height_m=self.height_m,
            force_total=self.force_total.copy(),
            moment_total=self.moment_total.copy(),
            thrust_N=self.thrust_N,
            gimbal_x=self.gimbal_x,
            gimbal_y=self.gimbal_y,
            gimbal_x_cmd=self.gimbal_x_cmd,
            gimbal_y_cmd=self.gimbal_y_cmd,
            servo_rate_x=self.servo_rate_x,
            servo_rate_y=self.servo_rate_y,
            servo_sat_x=self.servo_sat_x,
            servo_sat_y=self.servo_sat_y,
            force_thrust=self.force_thrust.copy(),
            force_gravity=self.force_gravity.copy(),
            thrust_dir_b=self.thrust_dir_b.copy(),
            modeled_thrust=self.modeled_thrust,
        )

    def sync_from_mass(self, mass_model) -> None:
        self.dry_mass_kg = float(getattr(mass_model, "dry_mass_kg", self.dry_mass_kg))
        self.fuel_kg = float(getattr(mass_model, "fuel_kg", self.fuel_kg))
        self.mass_kg = float(getattr(mass_model, "total_mass", lambda: self.dry_mass_kg + self.fuel_kg)())
