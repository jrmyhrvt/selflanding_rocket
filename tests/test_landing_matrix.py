"""
Invariant tests over the automated multi-scenario landing matrix.

These assert BEHAVIOUR (status vs. contact/speed/distance), not just that
classes exist. See evaluation/landing_matrix.py for the scenarios.
"""
from evaluation.landing_matrix import run_matrix, CASES
from evaluation import landing as L


ROWS = {r["name"]: r for r in run_matrix()}


def test_all_cases_run():
    assert len(ROWS) == len(CASES)


def test_landed_implies_valid_contact_and_speed():
    """Every LANDED case must really be on the ground, slow, and on target."""
    for r in ROWS.values():
        if r["status"] == L.LANDED:
            assert r["contact"] is True, r["name"]
            assert abs(r["height"]) <= L.H_TOL_M + 1e-9, r["name"]
            assert r["v_mag"] <= L.V_TOL_M_S + 1e-9, r["name"]
            assert r["dist_xy"] <= L.DIST_TOL_M + 1e-9, r["name"]


def test_nominal_cases_actually_land():
    """The guided scenarios must reach LANDED (non-trivial success)."""
    for name in ("nominal_center", "nominal_offset", "high_altitude",
                 "far_target", "low_start", "tight_dt"):
        assert ROWS[name]["status"] == L.LANDED, name


def test_timeout_is_not_a_landing():
    """Running out of simulation time while airborne must NOT be LANDED."""
    r = ROWS["timeout_short"]
    assert r["status"] != L.LANDED
    assert r["status"] == L.FAILED
    assert r["height"] > L.H_TOL_M  # still clearly airborne


def test_freefall_is_not_a_landing():
    r = ROWS["freefall_none"]
    assert r["status"] != L.LANDED


def test_hover_never_touches_down():
    r = ROWS["hover_no_touch"]
    assert r["contact"] is False
    assert r["status"] == L.FAILED


def test_tunnel_below_ground_is_not_a_landing():
    """A run past the ground (no stop_at_ground) must NOT count as LANDED."""
    r = ROWS["tunnel_no_stop"]
    assert r["height"] < -L.H_TOL_M      # ended well below ground
    assert r["status"] != L.LANDED
    assert r["status"] == L.FAILED
