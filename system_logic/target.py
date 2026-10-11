from dataclasses import dataclass
import numpy as np

@dataclass
class Target:
    pos: np.ndarray  # [x,y,z] inertial, Z up; landing on ground -> z near 0

    def distance_xy(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos[:2] - self.pos[:2]))

    def distance_3d(self, pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - self.pos))

    def v_mag(self, vel: np.ndarray) -> float:
        return float(np.linalg.norm(vel))
