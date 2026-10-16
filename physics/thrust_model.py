"""Simplified VIRTUAL gimballed-thrust model (learning simulation only).

This is NOT a real engine/thrust model. It is a deliberately coarse, tunable
mapping from a commanded thrust magnitude and two small nozzle gimbal angles to
a BODY-FRAME force and moment. Its only purpose is to give the unified 3D flight
view a physically consistent coupling between attitude and translation:

    gimbal -> body force F_b -> (rotated) inertial force -> translation
                            \\-> body moment M_b = r x F_b -> rotation

Conventions (matching the rest of the project):
- Body frame: rocket long axis is +Z (base at z=0, tip at z=+length).
- Frame: q is scalar-first [w,x,y,z] = q_{B->I}; body vectors are rotated to
  the world with `rotate_vector(q, v)`.
- The nozzle hinge sits below the base at r_b = [0, 0, -offset].
- Gimbal angles are small rotations of the thrust direction:
    pitch `gx` rotates about body +X, yaw `gy` about body +Y.
  Thrust direction (body frame):
    n_b = R_y(gy) @ R_x(gx) @ [0, 0, 1]
        = [cos(gx) sin(gy), -sin(gx), cos(gx) cos(gy)]
- Body force  F_b = thrust_N * n_b
- Body moment M_b = r_b x F_b = [offset*F_b_y, -offset*F_b_x, 0]
  (no roll authority about +Z: the thrust line passes through the hinge plane)
"""
import numpy as np


class ThrustModel:
    """A MODEL of a gimballed thruster. Not a real engine."""

    def __init__(self, max_thrust_N: float = 1000.0, gimbal_max_rad: float = 0.35,
                 gimbal_offset_m: float = 0.30):
        if max_thrust_N < 0.0:
            raise ValueError("max_thrust_N must be >= 0")
        if gimbal_max_rad < 0.0:
            raise ValueError("gimbal_max_rad must be >= 0")
        if gimbal_offset_m < 0.0:
            raise ValueError("gimbal_offset_m must be >= 0")
        self.max_thrust_N = float(max_thrust_N)
        self.gimbal_max_rad = float(gimbal_max_rad)
        self.gimbal_offset_m = float(gimbal_offset_m)

    def clamp(self, thrust_N: float, gx: float, gy: float):
        """Clamp thrust magnitude and gimbal angles to the model limits."""
        thrust_N = float(np.clip(thrust_N, 0.0, self.max_thrust_N))
        gx = float(np.clip(gx, -self.gimbal_max_rad, self.gimbal_max_rad))
        gy = float(np.clip(gy, -self.gimbal_max_rad, self.gimbal_max_rad))
        return thrust_N, gx, gy

    @staticmethod
    def thrust_direction_body(gx: float, gy: float) -> np.ndarray:
        """Unit thrust direction in the body frame for gimbal angles (gx, gy)."""
        return np.array([np.cos(gx) * np.sin(gy),
                         -np.sin(gx),
                         np.cos(gx) * np.cos(gy)])

    def force_moment(self, thrust_N: float, gx: float, gy: float):
        """Return (F_b[3], M_b[3], n_b[3]) for a commanded thrust + gimbal.

        thrust_N is clamped to [0, max_thrust_N]; angles to +-gimbal_max_rad.
        """
        thrust_N, gx, gy = self.clamp(thrust_N, gx, gy)
        n_b = self.thrust_direction_body(gx, gy)
        F_b = thrust_N * n_b
        r_b = np.array([0.0, 0.0, -self.gimbal_offset_m])
        M_b = np.cross(r_b, F_b)
        return F_b, M_b, n_b

    def gimbal_for_moment(self, M_b, thrust_N: float):
        """Inverse: body moment -> gimbal angles (best effort, clamped).

        Solves F_b = r^{-1} relation: F_b_y = M_x/offset, F_b_x = -M_y/offset,
        then maps the direction to the two gimbal angles. The commanded moment
        about +Z cannot be produced (always reported as 0 authority).
        """
        M_b = np.asarray(M_b, dtype=float)
        offset = max(self.gimbal_offset_m, 1e-9)
        thrust_N = float(np.clip(thrust_N, 1e-9, self.max_thrust_N))
        Fx = -M_b[1] / offset
        Fy = M_b[0] / offset
        # keep requested force within the available thrust budget
        mag_xy = float(np.hypot(Fx, Fy))
        if mag_xy > thrust_N:
            scale = thrust_N / mag_xy
            Fx *= scale
            Fy *= scale
        gx = float(np.arcsin(np.clip(-Fy / thrust_N, -1.0, 1.0)))
        cos_gx = max(np.cos(gx), 1e-9)
        gy = float(np.arcsin(np.clip((Fx / thrust_N) / cos_gx, -1.0, 1.0)))
        return self.clamp(thrust_N, gx, gy)[1:]
