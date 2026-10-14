import numpy as np
import pytest

from sensors.observed import SensorConfig, MeasurementModel, ObservedController
from estimator.base import SimpleEstimator, LowPassEstimator


class _State:
    def __init__(self, pos, vel, t=0.0):
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.t = t


class _RecordingController:
    """Inner controller stub that records the state it is given."""
    def __init__(self, cmd=None):
        self.seen = []
        self.cmd = np.zeros(3) if cmd is None else np.asarray(cmd, dtype=float)

    def compute(self, state, target_pos, mass_kg):
        self.seen.append((np.asarray(state.pos, dtype=float).copy(),
                          np.asarray(state.vel, dtype=float).copy()))
        return self.cmd


def test_sensor_config_validation():
    with pytest.raises(ValueError):
        SensorConfig(pos_std_m=-1.0).validate()
    with pytest.raises(ValueError):
        SensorConfig(dropout_prob=1.5).validate()
    with pytest.raises(ValueError):
        SensorConfig(dropout_prob=-0.1).validate()
    SensorConfig(pos_std_m=0.0, dropout_prob=0.0).validate()


def test_measurement_reproducible_with_seed():
    cfg = SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.2, seed=7)
    m1 = MeasurementModel(cfg)
    m2 = MeasurementModel(cfg)
    for t in np.arange(0.0, 1.0, 0.1):
        a = m1.sample([1, 2, 3], [0.1, 0.2, 0.3], t)
        b = m2.sample([1, 2, 3], [0.1, 0.2, 0.3], t)
        assert a.pos_valid == b.pos_valid
        assert a.vel_valid == b.vel_valid
        if a.pos is not None:
            assert np.allclose(a.pos, b.pos)


def test_noise_statistics_roughly_match_std():
    cfg = SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.0, seed=1)
    model = MeasurementModel(cfg)
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([0.0, 0.0, 0.0])
    samples = np.array([model.sample(pos, vel, t).pos for t in range(4000)])
    assert abs(samples.std() - 1.0) < 0.1
    vsamples = np.array([model.sample(pos, vel, t).vel for t in range(4000)])
    assert abs(vsamples.std() - 0.3) < 0.05


def test_bias_is_applied():
    cfg = SensorConfig(pos_std_m=0.0, vel_std_m_s=0.0,
                       pos_bias_m=2.0, vel_bias_m_s=-0.5, dropout_prob=0.0, seed=3)
    m = MeasurementModel(cfg).sample([0, 0, 0], [0, 0, 0], 0.0)
    assert np.allclose(m.pos, [2, 2, 2])
    assert np.allclose(m.vel, [-0.5, -0.5, -0.5])


def test_full_dropout_yields_no_samples():
    cfg = SensorConfig(pos_std_m=1.0, dropout_prob=1.0, seed=5)
    model = MeasurementModel(cfg)
    for t in range(50):
        m = model.sample([0, 0, 0], [0, 0, 0], t)
        assert m.pos is None and m.vel is None
        assert not m.pos_valid and not m.vel_valid


def test_zero_dropout_always_valid():
    cfg = SensorConfig(dropout_prob=0.0, seed=5)
    model = MeasurementModel(cfg)
    for t in range(50):
        m = model.sample([1, 1, 1], [0, 0, 0], t)
        assert m.pos_valid and m.vel_valid
        assert m.pos is not None and m.vel is not None


def test_lowpass_converges_to_constant_measurement():
    est = LowPassEstimator(alpha_pos=0.3, alpha_vel=0.3)
    target = np.array([5.0, -2.0, 1.0])
    for _ in range(200):
        out = est.update(target, target)
    assert np.allclose(out.pos, target, atol=1e-3)
    assert np.allclose(out.vel, target, atol=1e-3)


def test_lowpass_skips_missing_measurement():
    est = LowPassEstimator(alpha_pos=0.5, alpha_vel=0.5)
    est.update([10, 0, 0], [0, 0, 0])
    held = est.update(None, None)
    assert np.allclose(held.pos, [10, 0, 0])


def test_lowpass_alpha_bounds():
    with pytest.raises(ValueError):
        LowPassEstimator(alpha_pos=0.0)
    with pytest.raises(ValueError):
        LowPassEstimator(alpha_vel=1.5)


def test_simple_estimator_still_passthrough():
    est = SimpleEstimator()
    out = est.update([1, 2, 3], [4, 5, 6])
    assert np.allclose(out.pos, [1, 2, 3])
    assert np.allclose(out.vel, [4, 5, 6])


def test_observed_controller_does_not_mutate_true_state():
    cfg = SensorConfig(pos_std_m=0.0, vel_std_m_s=0.0, dropout_prob=0.0, seed=0)
    inner = _RecordingController()
    obs = ObservedController(inner, cfg)
    true_pos = np.array([3.0, 4.0, 5.0])
    true_vel = np.array([0.1, 0.0, -0.2])
    before_pos = true_pos.copy()
    before_vel = true_vel.copy()
    obs.compute(_State(true_pos, true_vel), np.zeros(3), 300.0)
    assert np.array_equal(true_pos, before_pos)
    assert np.array_equal(true_vel, before_vel)


def test_observed_controller_passes_estimate_not_truth():
    cfg = SensorConfig(pos_std_m=0.0, vel_std_m_s=0.0, dropout_prob=0.0, seed=0)
    inner = _RecordingController()
    obs = ObservedController(inner, cfg)
    obs.compute(_State([1, 2, 3], [0.1, 0.2, 0.3]), np.zeros(3), 300.0)
    seen_pos, seen_vel = inner.seen[-1]
    assert np.allclose(seen_pos, [1, 2, 3])
    assert np.allclose(seen_vel, [0.1, 0.2, 0.3])
    assert obs.samples_missing == 0


def test_observed_controller_full_dropout_does_not_crash():
    cfg = SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=1.0, seed=9)
    inner = _RecordingController(cmd=np.array([0.0, 0.0, 9.0]))
    obs = ObservedController(inner, cfg)
    cmd = obs.compute(_State([1, 2, 3], [0, 0, -1]), np.zeros(3), 300.0)
    assert cmd.shape == (3,)
    assert np.all(np.isfinite(cmd))
    assert obs.samples_missing == 1


def test_observed_controller_reports_rms_error():
    cfg = SensorConfig(pos_std_m=1.0, vel_std_m_s=0.3, dropout_prob=0.0, seed=4)
    inner = _RecordingController()
    obs = ObservedController(inner, cfg)
    for t in range(100):
        obs.compute(_State([0, 0, 0], [0, 0, 0], t), np.zeros(3), 300.0)
    epos, evel = obs.rms_errors()
    assert epos > 0.0 and evel > 0.0


def test_zero_noise_observed_matches_truth_inner():
    cfg = SensorConfig(pos_std_m=0.0, vel_std_m_s=0.0, dropout_prob=0.0, seed=0)
    inner = _RecordingController()
    obs = ObservedController(inner, cfg)
    for t in range(10):
        obs.compute(_State([t, 0, 0], [1, 0, 0], t), np.zeros(3), 300.0)
    assert np.allclose(inner.seen[-1][1], [1, 0, 0])
    assert obs.rms_errors() == (0.0, 0.0)
