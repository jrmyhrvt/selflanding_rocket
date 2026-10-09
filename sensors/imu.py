import numpy as np

class Accelerometer:
    def __init__(self, std_m_s2: float = 0.01, bias_m_s2: float = 0.0, dt_sample: float = 0.01):
        self.std = float(std_m_s2)
        self.bias = float(bias_m_s2)
        self.dt_sample = max(0.0, float(dt_sample))
        self._last_t = -1e18

    def measure(self, true_acc: np.ndarray, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.dt_sample > 0.0 and (t - self._last_t) < self.dt_sample - 1e-18:
            return true_acc + self.bias  # keep last? simple: return noisy once per sample; keep minimal
        self._last_t = t
        noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
        return true_acc + self.bias + noise


class Gyroscope:
    def __init__(self, std_rad_s: float = 0.001, bias_rad_s: float = 0.0, dt_sample: float = 0.01):
        self.std = float(std_rad_s)
        self.bias = float(bias_rad_s)
        self.dt_sample = max(0.0, float(dt_sample))
        self._last_t = -1e18

    def measure(self, true_omega: np.ndarray, t: float, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        if self.dt_sample > 0.0 and (t - self._last_t) < self.dt_sample - 1e-18:
            noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
            return true_omega + self.bias + noise
        self._last_t = t
        noise = rng.normal(0.0, self.std, size=3) if self.std > 0.0 else np.zeros(3)
        return true_omega + self.bias + noise
