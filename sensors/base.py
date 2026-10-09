import numpy as np

class Sensor:
    def __init__(self, name: str, std: float = 0.01, bias: float = 0.0, dt_sample: float = 0.01, delay_s: float = 0.0):
        self.name = name
        self.std = std
        self.bias = bias
        self.dt_sample = dt_sample
        self.delay_s = delay_s
        self._history = []

    def measure(self, true_value, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0.0, self.std) if self.std > 0.0 else 0.0
        meas = true_value + self.bias + noise
        # store (t, meas)
        self._history.append((t, meas))
        return meas
