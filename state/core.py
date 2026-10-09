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
    # mass
    mass_kg: float = 0.0
    fuel_kg: float = 0.0
    dry_mass_kg: float = 0.0
    # derived
    height_m: float = 0.0
    # forces/moments (optional)
    force_total: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment_total: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "State":
        return State(
            t=self.t,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            acc=self.acc.copy(),
            q=self.q.copy(),
            omega=self.omega.copy(),
            alpha=self.alpha.copy(),
            mass_kg=self.mass_kg,
            fuel_kg=self.fuel_kg,
            dry_mass_kg=self.dry_mass_kg,
            height_m=self.height_m,
            force_total=self.force_total.copy(),
            moment_total=self.moment_total.copy(),
        )
