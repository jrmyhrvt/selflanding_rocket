import numpy as np

class PositionSensor:
    def __init__(self, std_m: float = 1.0, bias_m: float = 0.0, dt_sample: float = 0.1):
        self.std = float(std_m)
        self.bias = float(bias_m)
        self.dt_sample = max(0.0, float(dt_sample))
        self._last_t = -1e18

    def measure(self, true_pos: np.ndarray, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.dt_sample > 0.0 and (t - self._last_t) < self.dt_sample - 1e-18:
            noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
            return true_pos + self.bias + noise
        self._last_t = t
        noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
        return true_pos + self.bias + noise


class VelocitySensor:
    def __init__(self, std_m_s: float = 0.1, bias_m_s: float = 0.0, dt_sample: float = 0.01):
        self.std = float(std_m_s)
        self.bias = float(bias_m_s)
        self.dt_sample = max(0.0, float(dt_sample))
        self._last_t = -1e18

    def measure(self, true_vel: np.ndarray, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.dt_sample > 0.0 and (t - self._last_t) < self.dt_sample - 1e-18:
            noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
            return true_vel + self.bias + noise
        self._last_t = t
        noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
        return true_vel + self.bias + noise
