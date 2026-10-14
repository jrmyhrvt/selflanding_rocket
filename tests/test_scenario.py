import numpy as np
import pytest

from simulation.scenario import Scenario, run_scenario
from evaluation import landing as L


def test_default_scenario_lands():
    hist = run_scenario(Scenario())
    assert len(hist) > 1
    res = L.evaluate_landing(hist, Scenario().target())
    assert res["status"] == L.LANDED


def test_initial_state_uses_settings():
    sc = Scenario(start_height_m=12.0, start_offset_xy=(2.0, -3.0),
                  v0_vertical_m_s=-1.5, v0_horizontal_m_s=0.8)
    st = sc.initial_state()
    assert np.allclose(st.pos, [2.0, -3.0, 12.0])
    assert np.allclose(st.vel, [0.8, 0.0, -1.5])


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        Scenario(dt=0.0).validate()
    with pytest.raises(ValueError):
        Scenario(dt=-0.1).validate()
    with pytest.raises(ValueError):
        Scenario(dt=2.0).validate()
    with pytest.raises(ValueError):
        Scenario(start_height_m=-1.0).validate()
    with pytest.raises(ValueError):
        Scenario(max_time_s=0.0).validate()
    # too many steps (tiny dt + long time)
    with pytest.raises(ValueError):
        Scenario(dt=0.001, max_time_s=300.0).validate()


def test_run_is_reproducible_and_fresh():
    sc = Scenario(start_height_m=15.0, target_xy=(2.0, 1.0), dt=0.05)
    h1 = run_scenario(sc)
    h2 = run_scenario(sc)
    assert h1 is not h2
    assert np.allclose(h1[-1].pos, h2[-1].pos)
    assert np.allclose(h1[-1].vel, h2[-1].vel)


def test_different_targets_change_trajectory():
    s1 = Scenario(start_height_m=15.0, target_xy=(4.0, 0.0), dt=0.05)
    s2 = Scenario(start_height_m=15.0, target_xy=(-4.0, 0.0), dt=0.05)
    h1 = run_scenario(s1)
    h2 = run_scenario(s2)
    assert not np.allclose(h1[-1].pos, h2[-1].pos)
    assert h1[-1].pos[0] > 0.0 and h2[-1].pos[0] < 0.0


def test_timeout_scenario_is_not_landed():
    sc = Scenario(start_height_m=60.0, max_time_s=2.0, dt=0.05)
    hist = run_scenario(sc)
    res = L.evaluate_landing(hist, sc.target())
    assert res["status"] == L.FAILED
