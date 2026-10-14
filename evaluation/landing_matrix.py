"""
Automated multi-scenario runner for the simplified autoland learning sim.

Runs several start states / target points through the SAME simulator +
controller + evaluator used by the app, and reports per case:
end status, end height, total speed and horizontal distance to target.

Run it directly for a human-readable table:

    PYTHONPATH=. python3 evaluation/landing_matrix.py

This is a learning model (point mass, no thrust limits). See docs.
"""
import numpy as np

from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from physics.controller import LandingController
from config import units
from evaluation import landing as L


class _Hover:
    """Diagnostic controller: cancels gravity, never touches down."""
    def compute(self, state, target_pos, mass_kg):
        return np.array([0.0, 0.0, units.GRAVITY_M_S2])


# Each case: name, h0, target (x,y), tf, dt, guidance kind, stop_at_ground.
CASES = [
    ("nominal_center", 25.0, (0.0, 0.0), 200.0, 0.05, "landing", True),
    ("nominal_offset", 25.0, (3.0, 2.0), 200.0, 0.05, "landing", True),
    ("high_altitude",  50.0, (2.0, 1.0), 300.0, 0.05, "landing", True),
    ("far_target",     25.0, (8.0, 0.0), 250.0, 0.05, "landing", True),
    ("low_start",       5.0, (1.0, 1.0), 120.0, 0.05, "landing", True),
    ("tight_dt",       20.0, (3.0, 2.0), 200.0, 0.02, "landing", True),
    ("timeout_short",  25.0, (3.0, 2.0),   5.0, 0.05, "landing", True),
    ("freefall_none",  50.0, (0.0, 0.0),   3.0, 0.01, None,      False),
    ("hover_no_touch", 30.0, (0.0, 0.0),   5.0, 0.01, "hover",   True),
    ("tunnel_no_stop", 25.0, (0.0, 0.0), 120.0, 0.05, "landing", False),
]


def run_case(case):
    name, h0, (tx, ty), tf, dt, kind, stop = case
    if kind == "landing":
        guidance = LandingController(descent_speed_m_s=0.5)
    elif kind == "hover":
        guidance = _Hover()
    else:
        guidance = None

    s = State(pos=np.array([0.0, 0.0, float(h0)]), vel=np.zeros(3))
    m = MassModel(dry_mass_kg=300.0, fuel_kg=0.0)
    cfg = SimConfig(tf=tf, dt=dt, stop_at_ground=bool(stop), ground_z=0.0)
    target = np.array([float(tx), float(ty), 0.0])
    sim = Simulator(cfg, guidance=guidance)
    hist = sim.run(s.copy(), m, target=target)

    res = L.evaluate_landing(hist, target)
    return {
        "name": name,
        "status": res["status"],
        "height": res["height"],
        "v_mag": res["v_mag"],
        "dist_xy": res["dist_xy"],
        "contact": res["contact"],
        "t_end": hist[-1].t if hist else None,
        "steps": len(hist),
        "guidance": kind or "none",
    }


def run_matrix(cases=None):
    return [run_case(c) for c in (cases or CASES)]


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}"


def main():
    rows = run_matrix()
    header = f"{'case':<16}{'status':<9}{'height':>9}{'v_mag':>9}{'dist_xy':>9}{'t_end':>9}{'steps':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<16}{r['status']:<9}{_fmt(r['height']):>9}"
              f"{_fmt(r['v_mag']):>9}{_fmt(r['dist_xy']):>9}"
              f"{_fmt(r['t_end']):>9}{r['steps']:>8}")
    return rows


if __name__ == "__main__":
    main()
