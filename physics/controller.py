import numpy as np
from config import units

G0 = units.GRAVITY_M_S2


class LandingController:
    """
    Simplified VIRTUAL guidance controller (learning model only).

    CONTRACT
    --------
    `compute(state, target_pos, mass_kg)` returns a THRUST acceleration
    command `a_cmd` (m/s^2) in the inertial frame, EXCLUDING gravity.
    The simulator adds gravity separately, so the total acceleration is:

        a_total = g + a_cmd        with g = [0, 0, -g0]

    Consequences of this contract:
      - To hover (a_total ~ 0) the controller must command a_cmd_z = +g0.
      - Gravity is never applied twice.

    MODEL (deliberately simplified)
    -------------------------------
      - Point mass; the mass argument does not change the command here.
      - Horizontal: proportional-derivative pull toward the target (X,Y).
      - Vertical: proportional control toward a target descent speed.
      - No thrust limits, no attitude/rotation, no engine dynamics.
      - This is NOT a realistic rocket/engine/autoland model.

    NOT for real hardware. Simulation only.
    """

    def __init__(self, descent_speed_m_s: float = 0.5, kp: float = 0.2,
                 kd: float = 0.4, kh: float = 1.5, horiz_ref_m: float = 5.0,
                 vxy_ref_m_s: float = 0.5):
        if descent_speed_m_s < 0.0:
            raise ValueError("descent_speed_m_s must be >= 0")
        self.descent_speed_m_s = float(descent_speed_m_s)
        self.kp = float(kp)
        self.kd = float(kd)
        self.kh = float(kh)
        # References for the descent gate (see below).
        self.horiz_ref_m = float(horiz_ref_m)
        self.vxy_ref_m_s = float(vxy_ref_m_s)

    def compute(self, state, target_pos, mass_kg: float) -> np.ndarray:
        pos = np.asarray(state.pos, dtype=float)
        vel = np.asarray(state.vel, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        # Horizontal PD toward the target point.
        err_xy = target_pos[:2] - pos[:2]
        ax = self.kp * err_xy[0] - self.kd * vel[0]
        ay = self.kp * err_xy[1] - self.kd * vel[1]

        # Horizontal descent gate: only descend at full speed once we are
        # roughly centred above the target AND horizontally slow. While far
        # away or still moving fast horizontally, hold altitude and translate
        # first, so horizontal velocity is damped before touchdown. This keeps
        # the TOTAL touchdown speed near the 0.5 m/s target.
        dist_xy = float(np.linalg.norm(err_xy))
        v_xy = float(np.linalg.norm(vel[:2]))
        gate = 1.0 / (1.0 + (dist_xy / self.horiz_ref_m) ** 2
                          + (v_xy / self.vxy_ref_m_s) ** 2)
        vz_des = -abs(self.descent_speed_m_s) * gate

        # a_total_des_z = kh * (vz_des - vz); convert to thrust command by
        # removing gravity:  a_cmd_z = a_total_des_z - g_z = ... + g0
        az = self.kh * (vz_des - vel[2]) + G0

        return np.array([ax, ay, az], dtype=float)
