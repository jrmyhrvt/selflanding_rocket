"""
Virtual sensing + observation layer for the UNIFIED 3D flight controller.

`ObservedController` (sensors/observed.py) is tailored to the point-mass
autoland controller (`compute(state, target_pos, mass_kg)`, position/velocity
only). The unified 3D controller (`simulation/flight.py::model_control`) also
reads the attitude quaternion `q` and the body angular rate `omega_b`, so it
needs a slightly wider observation layer. This module provides exactly that and
still reuses the SAME existing sensor classes:

    MeasurementModel  (sensors/observed.py)  -> position + velocity (+ dropout)
    Gyroscope         (sensors/imu.py)       -> body angular rate
    LowPassEstimator  (estimator/base.py)    -> optional estimate

Three things stay strictly separate (same rule as sensors/observed.py):

    TRUE state   : the State integrated by simulation/flight.py
    MEASUREMENT  : the simulated sensor value (noise, bias, dropouts)
    ESTIMATE     : what the controller actually uses

Documented simplification: there is no ATTITUDE sensor in this learning model,
so the observed state forwards the TRUE orientation `q`. Only position, velocity
and body rate are measured/noisy; a real attitude estimate is out of scope.

Reproducibility: the noise stream comes from `MeasurementModel`, seeded from
`SensorConfig.seed`; identical configs yield identical data.

Simplified learning model only. Not a realistic sensor/estimation model.
"""
import numpy as np

from sensors.observed import MeasurementModel, SensorConfig
from sensors.imu import Gyroscope


class ObservedFlightController:
    """Drop-in controller for `simulation.flight.run_flight(controller=...)`.

    Wraps an inner command function `inner(state, m, thr, cfg)` (typically
    `simulation.flight.model_control`) and hands it an OBSERVED state built from
    measurements/estimates instead of the true state. The true state is only
    READ to synthesise measurements and to score the estimate; it never steers
    directly.
    """

    def __init__(self, inner, cfg: SensorConfig, gyro_std_rad_s: float = 0.0,
                 estimator=None):
        cfg.validate()
        self.inner = inner
        self.cfg = cfg
        self.model = MeasurementModel(cfg)
        self.gyro = Gyroscope(std_rad_s=gyro_std_rad_s, bias_rad_s=0.0,
                              dt_sample=0.0)
        self.estimator = estimator
        self._hold_pos = np.zeros(3)
        self._hold_vel = np.zeros(3)
        self.samples_missing = 0
        self.n = 0
        self.err_pos_sq = 0.0
        self.err_vel_sq = 0.0

    def __call__(self, state, m, thr, cfg):
        meas = self.model.sample(state.pos, state.vel, state.t)
        omega_meas = np.asarray(
            self.gyro.measure(np.asarray(state.omega_b, dtype=float), state.t,
                              rng=self.model.rng), dtype=float)

        if (not meas.pos_valid) or (not meas.vel_valid):
            self.samples_missing += 1
        if meas.pos is not None:
            self._hold_pos = np.asarray(meas.pos, dtype=float)
        if meas.vel is not None:
            self._hold_vel = np.asarray(meas.vel, dtype=float)

        if self.estimator is not None:
            est = self.estimator.update(meas.pos, meas.vel)
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

        # Observed state: true orientation (no attitude sensor modelled),
        # measured body rate, estimated/held position and velocity.
        obs = state.copy()
        obs.pos = est_pos
        obs.vel = est_vel
        obs.omega_b = omega_meas
        return self.inner(obs, m, thr, cfg)

    def rms_errors(self):
        if self.n == 0:
            return None, None
        return (self.err_pos_sq / self.n) ** 0.5, (self.err_vel_sq / self.n) ** 0.5
