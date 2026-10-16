"""Virtual rigid body model for rotational dynamics (learning simulation).

Conventions (documented):
- Inertial (I): ENU (X East, Y North, Z Up)
- Body (B): body-fixed frame
- q is scalar-first [w, x, y, z] representing orientation q_{B->I}
- omega_b, alpha_b, M_b are expressed in the body frame (standard Euler equations)
- I_b is inertia tensor in body frame; here simplified as diagonal I_diag = [Ixx, Iyy, Izz]
"""
import numpy as np


class RigidBody:
    def __init__(self, mass_kg=1.0, I_diag=None):
        if I_diag is None:
            I_diag = [1.0, 1.0, 1.0]
        self.mass_kg = float(mass_kg)
        self.I_diag = np.asarray(I_diag, dtype=float)
        if self.I_diag.shape != (3,):
            raise ValueError("I_diag must be (3,)")
        if np.any(self.I_diag <= 0.0):
            raise ValueError("I_diag components must be > 0")
        self.I = np.diag(self.I_diag)
        self.I_inv = np.diag(1.0 / self.I_diag)

    def inertia(self):
        return self.I.copy()

    def inv_inertia(self):
        return self.I_inv.copy()

    def angular_accel(self, omega_b, M_b):
        """
        Compute body-frame angular acceleration alpha_b from Euler's equations.
        alpha_b = I_b^{-1} (M_b - omega_b × (I_b * omega_b))
        """
        omega_b = np.asarray(omega_b, dtype=float)
        M_b = np.asarray(M_b, dtype=float)
        if omega_b.shape != (3,) or M_b.shape != (3,):
            raise ValueError("omega_b and M_b must be (3,)")
        h = self.I @ omega_b
        gyro = np.cross(omega_b, h)
        alpha_b = self.I_inv @ (M_b - gyro)
        return alpha_b
