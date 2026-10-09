import numpy as np

class Sensor:
    def __init__(self, name: str, std: float = 0.01, bias: float = 0.0, dt_sample: float = 0.01, delay_s: float = 0.0):
        self.name = name
        self.std = float(std)
        self.bias = float(bias)
        self.dt_sample = max(0.0, float(dt_sample))
        self.delay_s = max(0.0, float(delay_s))
        self._history = []

    def measure(self, true_value, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0.0, self.std) if self.std > 0.0 else 0.0
        meas = true_value + self.bias + noise
        self._history.append((float(t), meas))
        return meas
