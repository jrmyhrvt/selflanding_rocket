import numpy as np

# Default evaluation thresholds (documented, configurable).
V_TOL_M_S = 0.5       # maximum total speed for a soft landing
H_TOL_M = 0.05        # vertical tolerance for "ground contact"
DIST_TOL_M = 1.0      # horizontal tolerance to the target point

LANDED = "LANDED"
TOO_FAST = "TOO_FAST"
FAILED = "FAILED"
NO_DATA = "NO_DATA"


def evaluate_landing(history, target_pos, ground_z: float = 0.0,
                     v_tol: float = V_TOL_M_S, h_tol: float = H_TOL_M,
                     dist_tol: float = DIST_TOL_M) -> dict:
    """
    Evaluate a landing run from a list of State objects.

    Criteria (deliberately simple and explicit):
      - Ground contact: final height <= ground_z + h_tol.
      - LANDED   : contact AND total speed <= v_tol AND horizontal
                   distance to target <= dist_tol.
      - TOO_FAST : contact but total speed > v_tol.
      - FAILED   : no ground contact by the end, OR contact but the
                   landing point is off-target (horizontal dist > dist_tol).

    Returns a dict with status, final values and the thresholds used.
    Does not fabricate a landing: an airborne run is never LANDED.
    """
    if not history:
        return {
            "status": NO_DATA,
            "v_mag": None, "dist_xy": None, "height": None,
            "v_tol": v_tol, "h_tol": h_tol, "dist_tol": dist_tol,
        }

    target_pos = np.asarray(target_pos, dtype=float)
    hf = history[-1]
    pos = np.asarray(hf.pos, dtype=float)
    vel = np.asarray(hf.vel, dtype=float)

    height = float(pos[2])
    v_mag = float(np.linalg.norm(vel))
    dist_xy = float(np.linalg.norm(pos[:2] - target_pos[:2]))
    # Valid ground contact requires the final height to be within a band
    # around the ground plane. A state far BELOW ground means the trajectory
    # tunnelled through the floor / simulation ran past the ground and must
    # NOT be counted as a successful touchdown.
    contact = abs(height - ground_z) <= h_tol

    # Small floating-point tolerance so a nominal 0.5 m/s landing is not
    # misclassified as TOO_FAST by numerical noise.
    v_eps = 1e-9
    d_eps = 1e-9

    if not contact:
        status = FAILED
    elif v_mag > v_tol + v_eps:
        status = TOO_FAST
    elif dist_xy > dist_tol + d_eps:
        status = FAILED
    else:
        status = LANDED

    return {
        "status": status,
        "v_mag": v_mag,
        "dist_xy": dist_xy,
        "height": height,
        "contact": bool(contact),
        "v_tol": v_tol,
        "h_tol": h_tol,
        "dist_tol": dist_tol,
    }
