from dataclasses import dataclass
import numpy as np
from state.core import State
from physics.translation import Translation
from atmosphere import isa
from mass_model.mass import MassModel
from config import units

@dataclass
class SimConfig:
    t0: float = 0.0
    tf: float = 10.0
    dt: float = units.DEFAULT_DT
    seed: int = 12345

class Simulator:
    def __init__(self, cfg: SimConfig = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.trans = Translation(self.cfg.dt)

    def run(self, state: State, mass: MassModel, n_steps=None):
        if n_steps is None:
            n_steps = int((self.cfg.tf - self.cfg.t0) / self.cfg.dt)
        history = []
        for i in range(n_steps):
            t = self.cfg.t0 + i * self.cfg.dt
            state.t = t
            # simple gravity
            rho = isa.density(state.height_m)
            g = np.array([0.0, 0.0, -units.GRAVITY_M_S2])
            forces = mass.total_mass() * g
            self.trans.step(state, forces, mass.total_mass())
            history.append(state.copy())
        return history
