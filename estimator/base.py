from dataclasses import dataclass
import numpy as np

@dataclass
class Estimate:
    pos: np.ndarray
    vel: np.ndarray
    cov: np.ndarray  # placeholder

class SimpleEstimator:
    def __init__(self):
        self.x = np.zeros(6)

    def update(self, z_pos, z_vel):
        self.x[0:3] = z_pos
        self.x[3:6] = z_vel
        return Estimate(self.x[0:3].copy(), self.x[3:6].copy(), np.eye(6)*1e-3)
