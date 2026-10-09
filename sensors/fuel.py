import numpy as np

class FuelSensor:
    def __init__(self, std_kg: float = 2.0, bias_kg: float = 0.0, dt_sample: float = 1.0):
        self.std = float(std_kg)
        self.bias = float(bias_kg)
        self.dt_sample = max(0.0, float(dt_sample))
        self._last_t = -1e18

    def measure(self, true_fuel_kg: float, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.dt_sample > 0.0 and (t - self._last_t) < self.dt_sample - 1e-18:
            noise = rng.normal(0.0, self.std) if self.std > 0.0 else 0.0
            m = true_fuel_kg + self.bias + noise
            if m < 0.0:
                m = 0.0
            return m
        self._last_t = t
        noise = rng.normal(0.0, self.std) if self.std > 0.0 else 0.0
        m = true_fuel_kg + self.bias + noise
        if m < 0.0:
            m = 0.0
        return m
