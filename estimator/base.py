from dataclasses import dataclass
import numpy as np

@dataclass
class Estimate:
    pos: np.ndarray
    vel: np.ndarray
    cov: np.ndarray  # placeholder

class SimpleEstimator:
    """Trivial passthrough estimator (no filtering, no fusion).

    It simply copies the measurement into the estimate. With noisy
    measurements the estimate is therefore as noisy as the measurement.
    Requires a measurement every call (no None handling).
    """
    def __init__(self):
        self.x = np.zeros(6)

    def update(self, z_pos, z_vel):
        self.x[0:3] = z_pos
        self.x[3:6] = z_vel
        return Estimate(self.x[0:3].copy(), self.x[3:6].copy(), np.eye(6)*1e-3)


class LowPassEstimator:
    """External, simplified first-order low-pass estimator (learning model).

        x <- x + alpha * (z - x)

    - alpha in (0, 1]; alpha=1 behaves like the passthrough estimator.
    - A missing measurement (None) is skipped: the last estimate is kept.
    - Deliberately simple; NOT a Kalman filter and no covariance logic.
    """

    def __init__(self, alpha_pos: float = 0.25, alpha_vel: float = 0.25):
        if not (0.0 < alpha_pos <= 1.0) or not (0.0 < alpha_vel <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha_pos = float(alpha_pos)
        self.alpha_vel = float(alpha_vel)
        self.x_pos = None
        self.x_vel = None

    def update(self, z_pos, z_vel):
        if z_pos is not None:
            z_pos = np.asarray(z_pos, dtype=float)
            if self.x_pos is None:
                self.x_pos = z_pos.copy()
            else:
                self.x_pos = self.x_pos + self.alpha_pos * (z_pos - self.x_pos)
        if z_vel is not None:
            z_vel = np.asarray(z_vel, dtype=float)
            if self.x_vel is None:
                self.x_vel = z_vel.copy()
            else:
                self.x_vel = self.x_vel + self.alpha_vel * (z_vel - self.x_vel)
        pos = self.x_pos.copy() if self.x_pos is not None else np.zeros(3)
        vel = self.x_vel.copy() if self.x_vel is not None else np.zeros(3)
        return Estimate(pos, vel, np.eye(6) * 1e-3)
