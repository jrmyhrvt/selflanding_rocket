import numpy as np

class Translation:
    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def step(self, state, forces: np.ndarray, mass_kg: float):
        if mass_kg <= 1e-12:
            acc = np.zeros(3)
        else:
            acc = forces / mass_kg
        state.acc = acc
        state.vel = state.vel + acc * self.dt
        state.pos = state.pos + state.vel * self.dt
        state.height_m = state.pos[2] if len(state.pos) > 2 else 0.0
        state.force_total = forces.copy()
        return state
