from dataclasses import dataclass
import numpy as np
from state.core import State
from physics.translation import Translation
try:
    from physics.rotation import integrate_rotation, integrate_rotation_body
except Exception:  # pragma: no cover
    def integrate_rotation(state, dt):
        return state

    def integrate_rotation_body(state, dt, rb=None):
        return state
try:
    from physics.rigidbody import RigidBody
except Exception:  # pragma: no cover
    RigidBody = None
from atmosphere import isa
from mass_model.mass import MassModel
from config import units
from system_logic.monitor import Monitor


@dataclass
class SimConfig:
    t0: float = 0.0
    tf: float = 10.0
    dt: float = units.DEFAULT_DT
    seed: int = 12345
    gravity: bool = True
    # Ground handling is OPT-IN so the default behaviour stays unchanged.
    stop_at_ground: bool = False
    ground_z: float = 0.0
    # Rotation integration is OPT-IN (default False) to preserve existing behavior.
    integrate_rotation: bool = False
    integrate_rotation_body: bool = False
    rigid_body: object = None


class Simulator:
    """
    Simplified point-mass simulator (learning model).

    Physics contract:
      - Forces are summed in the inertial frame and divided by total mass.
      - Gravity (if enabled) is `F_g = m * [0, 0, -g0]`.
      - An OPTIONAL guidance controller may supply a THRUST acceleration
        command `a_cmd` (m/s^2) WITHOUT gravity. The simulator adds it as
        `F_guidance = m * a_cmd`. Total acceleration is therefore
        `a_total = g + a_cmd`.
      - When `guidance is None` the behaviour is exactly the historical one
        (gravity only), so existing tests/behaviour are preserved.

    Limitations: point mass, no thrust limits, no attitude/rotation coupling
      in the legacy sense; rotation can be integrated optionally.
    NOT a realistic rocket model.
    """

    def __init__(self, cfg: SimConfig = None, monitor: Monitor = None, guidance=None):
        self.cfg = cfg or SimConfig()
        if self.cfg.dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.rng = np.random.default_rng(self.cfg.seed)
        self.trans = Translation(self.cfg.dt)
        self.monitor = monitor
        self.guidance = guidance

    def _gravity_force(self, mass_kg: float) -> np.ndarray:
        if not self.cfg.gravity:
            return np.zeros(3)
        g = np.array([0.0, 0.0, -units.GRAVITY_M_S2])
        return mass_kg * g

    def run(self, state: State, mass: MassModel, n_steps=None, target=None):
        if n_steps is None:
            n_steps = int((self.cfg.tf - self.cfg.t0) / self.cfg.dt)
        history = []
        for i in range(n_steps):
            t = self.cfg.t0 + i * self.cfg.dt
            state.t = t
            state.sync_from_mass(mass)

            forces = self._gravity_force(mass.total_mass())
            if self.guidance is not None and target is not None:
                a_cmd = np.asarray(self.guidance.compute(state, target, mass.total_mass()), dtype=float)
                if a_cmd.shape != (3,):
                    raise ValueError("guidance.compute must return a 3-vector (m/s^2)")
                forces = forces + mass.total_mass() * a_cmd

            self.trans.step(state, forces, mass.total_mass())

            # Ensure at most one rotation integrator is active (opt-in).
            # If both are True, the body-frame path takes precedence.
            rot_legacy = getattr(self.cfg, "integrate_rotation", False)
            rot_body = getattr(self.cfg, "integrate_rotation_body", False)
            if rot_body:
                rb = getattr(self.cfg, "rigid_body", None)
                integrate_rotation_body(state, self.cfg.dt, rb=rb)
            elif rot_legacy:
                integrate_rotation(state, self.cfg.dt)

            if self.monitor is not None:
                self.monitor.check_plausible(state)

            # Optional ground contact handling (opt-in; default off).
            if self.cfg.stop_at_ground and state.pos[2] <= self.cfg.ground_z:
                state.pos[2] = self.cfg.ground_z
                state.height_m = self.cfg.ground_z
                history.append(state.copy())
                break

            history.append(state.copy())
        return history
