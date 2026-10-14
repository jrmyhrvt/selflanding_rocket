"""
Virtual sensing + observation layer for the learning sim.

This module keeps three things strictly separate:

    TRUE state      : the State integrated by simulation/loop.py
    MEASUREMENT     : the simulated sensor value (noise, bias, dropouts)
    ESTIMATE        : what the controller actually uses

Rules:
  - A measurement NEVER modifies the true state. The true state is only READ
    (a copy is taken) to synthesise a measurement.
  - The controller receives the ESTIMATE, never the true state.

`ObservedController` is a drop-in guidance object for
`simulation.scenario.run_scenario_with`; it wraps an inner controller (e.g.
`physics.controller.LandingController`) so the physics code is reused
unchanged.

Reproducibility: every random draw comes from one `numpy.random.default_rng`
seeded from `SensorConfig.seed`, so identical configs yield identical data.

Simplified learning model only. Not a realistic sensor/estimation model.
"""
from dataclasses import dataclass
import numpy as np

from sensors.posvel import PositionSensor, VelocitySensor


@dataclass
class SensorConfig:
    """Documented, purely virtual sensor parameters."""
    pos_std_m: float = 1.0          # position noise std per axis [m]
    vel_std_m_s: float = 0.3        # velocity noise std per axis [m/s]
    pos_bias_m: float = 0.0         # constant position bias [m]
    vel_bias_m_s: float = 0.0       # constant velocity bias [m/s]
    dropout_prob: float = 0.0       # probability that a sample is missing
    seed: int = 1234

    def validate(self) -> None:
        if self.pos_std_m < 0.0 or self.vel_std_m_s < 0.0:
            raise ValueError("sensor noise std must be >= 0")
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1]")


@dataclass
class Measurement:
    """One simulated sample. `None` means the value is missing (dropout)."""
    pos: object      # np.ndarray(3) or None
    vel: object      # np.ndarray(3) or None
    pos_valid: bool
    vel_valid: bool


class MeasurementModel:
    def __init__(self, cfg: SensorConfig):
        cfg.validate()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.pos_sensor = PositionSensor(std_m=cfg.pos_std_m, bias_m=cfg.pos_bias_m,
                                         dt_sample=0.0)
        self.vel_sensor = VelocitySensor(std_m_s=cfg.vel_std_m_s, bias_m_s=cfg.vel_bias_m_s,
                                         dt_sample=0.0)

    def sample(self, true_pos, true_vel, t: float) -> Measurement:
        """Synthesise a measurement from a COPY of the true values."""
        true_pos = np.asarray(true_pos, dtype=float)
        true_vel = np.asarray(true_vel, dtype=float)
        pos = self.pos_sensor.measure(true_pos, t, rng=self.rng)
        vel = self.vel_sensor.measure(true_vel, t, rng=self.rng)
        # Dropouts: independent per channel; reproducible via self.rng.
        pos_valid = bool(self.rng.random() >= self.cfg.dropout_prob)
        vel_valid = bool(self.rng.random() >= self.cfg.dropout_prob)
        return Measurement(
            pos=pos if pos_valid else None,
            vel=vel if vel_valid else None,
            pos_valid=pos_valid,
            vel_valid=vel_valid,
        )


class _EstState:
    """Minimal duck-typed state handed to the inner controller."""
    __slots__ = ("pos", "vel", "t")

    def __init__(self, pos, vel, t):
        self.pos = pos
        self.vel = vel
        self.t = t


class ObservedController:
    """
    Guidance wrapper that converts the true state into measurements and
    passes only the ESTIMATE to the inner controller.

    Sampling policy for missing data (documented, deliberately simple):
      - no estimator: keep the last valid measurement (sample-and-hold).
      - with estimator: forward `None` so the estimator can skip/hold.
      - before the first valid sample the held value is zeros.

    Scoring (RMS error vs. true state) only READS the true state; it is
    never used to steer.
    """

    def __init__(self, inner, cfg: SensorConfig, estimator=None,
                 record_series: bool = False):
        self.inner = inner
        self.model = MeasurementModel(cfg)
        self.estimator = estimator
        self._hold_pos = np.zeros(3)
        self._hold_vel = np.zeros(3)
        self.samples_missing = 0
        self.n = 0
        self.err_pos_sq = 0.0
        self.err_vel_sq = 0.0
        # Optional per-step recording of the TRUE state, the MEASUREMENT and
        # the ESTIMATE. Used for diagnostics. Recording only appends data and
        # never changes the RNG stream or the simulation, so a recorded run is
        # numerically identical to an unrecorded one.
        self.record_series = bool(record_series)
        self.record = []

    def compute(self, state, target_pos, mass_kg):
        m = self.model.sample(state.pos, state.vel, state.t)
        if (not m.pos_valid) or (not m.vel_valid):
            self.samples_missing += 1

        if m.pos is not None:
            self._hold_pos = np.asarray(m.pos, dtype=float)
        if m.vel is not None:
            self._hold_vel = np.asarray(m.vel, dtype=float)

        if self.estimator is not None:
            est = self.estimator.update(m.pos, m.vel)  # None signals "missing"
            est_pos = np.asarray(est.pos, dtype=float)
            est_vel = np.asarray(est.vel, dtype=float)
        else:
            est_pos = self._hold_pos.copy()
            est_vel = self._hold_vel.copy()

        # Scoring only (reads truth, never steers).
        dpos = est_pos - np.asarray(state.pos, dtype=float)
        dvel = est_vel - np.asarray(state.vel, dtype=float)
        self.err_pos_sq += float(np.dot(dpos, dpos))
        self.err_vel_sq += float(np.dot(dvel, dvel))
        self.n += 1

        if self.record_series:
            self.record.append({
                "t": float(state.t),
                "pos_true": np.asarray(state.pos, dtype=float).copy(),
                "vel_true": np.asarray(state.vel, dtype=float).copy(),
                "pos_meas": None if m.pos is None else np.asarray(m.pos, dtype=float).copy(),
                "vel_meas": None if m.vel is None else np.asarray(m.vel, dtype=float).copy(),
                "pos_est": est_pos.copy(),
                "vel_est": est_vel.copy(),
                "pos_valid": m.pos_valid,
                "vel_valid": m.vel_valid,
            })

        return self.inner.compute(_EstState(est_pos, est_vel, state.t),
                                  target_pos, mass_kg)

    def rms_errors(self):
        if self.n == 0:
            return None, None
        return (self.err_pos_sq / self.n) ** 0.5, (self.err_vel_sq / self.n) ** 0.5
