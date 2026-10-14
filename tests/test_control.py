import numpy as np
from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from system_logic.target import Target
from physics.controller import LandingController
from evaluation import landing as landing_eval
from config import units

G0 = units.GRAVITY_M_S2


def _free_fall_history(tf=1.0, dt=0.01, h0=100.0, z0=0.0, v0=(0.0, 0.0, 0.0)):
    s = State(pos=np.array([0.0, 0.0, h0]), vel=np.array(v0, dtype=float))
    m = MassModel(500.0, 0.0)
    sim = Simulator(SimConfig(tf=tf, dt=dt, seed=1))
    return sim.run(s.copy(), m)


def test_without_guidance_behaviour_unchanged():
    """No controller => pure gravity, exactly as before."""
    hist = _free_fall_history(tf=1.0, dt=0.01, h0=100.0)
    # acc == -g (gravity) at every step, no guidance contribution
    for h in hist:
        assert np.allclose(h.acc, np.array([0.0, 0.0, -G0]), atol=1e-9)


def test_guidance_none_equals_default():
    """Passing guidance=None must not change the result at all."""
    s1 = State(pos=np.array([0.0, 0.0, 80.0]))
    s2 = State(pos=np.array([0.0, 0.0, 80.0]))
    m1 = MassModel(300.0, 0.0)
    m2 = MassModel(300.0, 0.0)
    h1 = Simulator(SimConfig(tf=1.0, dt=0.01)).run(s1.copy(), m1)
    h2 = Simulator(SimConfig(tf=1.0, dt=0.01), guidance=None).run(s2.copy(), m2)
    assert np.allclose(h1[-1].pos, h2[-1].pos)
    assert np.allclose(h1[-1].vel, h2[-1].vel)


def test_hover_controller_prevents_fall():
    """A controller commanding a_cmd_z = +g0 exactly cancels gravity -> hover."""
    class HoverController:
        def compute(self, state, target_pos, mass_kg):
            return np.array([0.0, 0.0, G0])

    s = State(pos=np.array([0.0, 0.0, 50.0]))
    m = MassModel(200.0, 0.0)
    sim = Simulator(SimConfig(tf=2.0, dt=0.01), guidance=HoverController())
    hist = sim.run(s.copy(), m, target=np.array([0.0, 0.0, 0.0]))

    # With gravity fully cancelled the vertical velocity stays ~0.
    assert abs(hist[-1].vel[2]) < 1e-6
    assert abs(hist[-1].height_m - 50.0) < 1e-6
    # And it is NOT free fall (would have dropped ~20 m).
    assert hist[-1].height_m > 49.0


def test_controller_output_changes_motion():
    """Fresh controller output must visibly affect the simulated motion."""
    s_plain = State(pos=np.array([0.0, 0.0, 60.0]))
    s_ctrl = State(pos=np.array([0.0, 0.0, 60.0]))
    m1 = MassModel(300.0, 0.0)
    m2 = MassModel(300.0, 0.0)

    # Plain (gravity only) vs. guided (controller commands +g)
    h_plain = Simulator(SimConfig(tf=1.0, dt=0.01)).run(s_plain.copy(), m1)

    class HoverController:
        def compute(self, state, target_pos, mass_kg):
            return np.array([0.0, 0.0, G0])

    h_ctrl = Simulator(SimConfig(tf=1.0, dt=0.01), guidance=HoverController()).run(
        s_ctrl.copy(), m2, target=np.array([0.0, 0.0, 0.0])
    )
    assert h_plain[-1].height_m < h_ctrl[-1].height_m - 1.0


def test_controller_sign_pulls_toward_target_xy():
    """Horizontal PD must move the rocket toward an offset target."""
    controller = LandingController(descent_speed_m_s=0.0, kp=0.3, kd=0.1, kh=0.6)
    s = State(pos=np.array([0.0, 0.0, 40.0]), vel=np.zeros(3))
    m = MassModel(250.0, 0.0)
    target = np.array([5.0, -3.0, 0.0])
    sim = Simulator(SimConfig(tf=3.0, dt=0.005), guidance=controller)
    hist = sim.run(s.copy(), m, target=target)

    start_dist = np.linalg.norm(hist[0].pos[:2] - target[:2])
    end_dist = np.linalg.norm(hist[-1].pos[:2] - target[:2])
    assert end_dist < start_dist
    # moved in +x and -y toward the target
    assert hist[-1].pos[0] > 0.0
    assert hist[-1].pos[1] < 0.0


def test_various_targets():
    """Different targets move the rocket in the matching direction."""
    controller = LandingController(descent_speed_m_s=0.0)
    for tx, ty in [(2.0, 0.0), (0.0, 2.0), (-2.0, -1.0)]:
        s = State(pos=np.array([0.0, 0.0, 30.0]))
        m = MassModel(250.0, 0.0)
        sim = Simulator(SimConfig(tf=2.0, dt=0.005), guidance=controller)
        hist = sim.run(s.copy(), m, target=np.array([tx, ty, 0.0]))
        dx = hist[-1].pos[0] - 0.0
        dy = hist[-1].pos[1] - 0.0
        if tx > 0:
            assert dx > 0
        elif tx < 0:
            assert dx < 0
        if ty > 0:
            assert dy > 0
        elif ty < 0:
            assert dy < 0


def test_guided_run_reaches_landed():
    """A coupled guided run must actually satisfy the LANDED criteria."""
    ctrl = LandingController(descent_speed_m_s=0.5)
    s = State(pos=np.array([0.0, 0.0, 20.0]))
    m = MassModel(300.0, 0.0)
    cfg = SimConfig(tf=200.0, dt=0.05, stop_at_ground=True, ground_z=0.0)
    target = np.array([3.0, 2.0, 0.0])
    hist = Simulator(cfg, guidance=ctrl).run(s.copy(), m, target=target)
    res = landing_eval.evaluate_landing(hist, target)
    assert res["status"] == landing_eval.LANDED
    assert res["v_mag"] <= landing_eval.V_TOL_M_S
    assert res["dist_xy"] <= landing_eval.DIST_TOL_M


def test_free_fall_does_not_land():
    """Without guidance the rocket is not evaluated as LANDED (no soft touchdown)."""
    s = State(pos=np.array([0.0, 0.0, 50.0]))
    m = MassModel(300.0, 0.0)
    cfg = SimConfig(tf=3.0, dt=0.01)  # no stop_at_ground: keeps falling
    hist = Simulator(cfg).run(s.copy(), m)
    res = landing_eval.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]))
    assert res["status"] != landing_eval.LANDED


def test_hover_controller_not_falsely_landed():
    """A controller that just hovers never touches down -> not LANDED."""
    class HoverController:
        def compute(self, state, target_pos, mass_kg):
            return np.array([0.0, 0.0, G0])

    s = State(pos=np.array([0.0, 0.0, 30.0]))
    m = MassModel(200.0, 0.0)
    cfg = SimConfig(tf=5.0, dt=0.01, stop_at_ground=True, ground_z=0.0)
    hist = Simulator(cfg, guidance=HoverController()).run(
        s.copy(), m, target=np.array([0.0, 0.0, 0.0])
    )
    res = landing_eval.evaluate_landing(hist, np.array([0.0, 0.0, 0.0]))
    assert res["contact"] is False
    assert res["status"] == landing_eval.FAILED


def test_ground_contact_clamps_height():
    """With stop_at_ground the run ends at the ground plane."""
    class HoverNone:
        def compute(self, state, target_pos, mass_kg):
            return np.array([0.0, 0.0, G0])  # hover-ish; then we force descent

    s = State(pos=np.array([0.0, 0.0, 3.0]))
    m = MassModel(100.0, 0.0)
    cfg = SimConfig(tf=5.0, dt=0.01, stop_at_ground=True, ground_z=0.0)
    hist = Simulator(cfg).run(s.copy(), m)  # gravity only, will hit ground
    assert abs(hist[-1].pos[2] - 0.0) < 1e-9
    assert abs(hist[-1].height_m) < 1e-9
    # Should stop early, not run the full tf.
    assert len(hist) < 500
