"""
Scenario definition + runner for the simplified autoland learning sim.

This module only ORCHESTRATES existing components:

    State          (state/core.py)
    MassModel      (mass_model/mass.py)
    Simulator      (simulation/loop.py)   <- physics
    LandingController (physics/controller.py) <- control
    evaluate_landing  (evaluation/landing.py) <- assessment

It contains NO parallel physics/control logic; it just assembles the pieces
so the app and the tests use exactly one code path.

Learning model only (point mass, no thrust limits). NOT for real hardware.
"""
from dataclasses import dataclass, field
import numpy as np

from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from physics.controller import LandingController
from physics.rigidbody import RigidBody


DEFAULT_START_HEIGHT_M = 25.0
DEFAULT_START_OFFSET_XY = (0.0, 0.0)
DEFAULT_TARGET_XY = (3.0, 2.0)
DEFAULT_V0_VERTICAL_M_S = 0.0
DEFAULT_V0_HORIZONTAL_M_S = 0.0
DEFAULT_DT_S = 0.05
DEFAULT_MAX_TIME_S = 300.0
DEFAULT_DESCENT_SPEED_M_S = 0.5
DEFAULT_DRY_MASS_KG = 300.0

# Safety guard: keep the number of integration steps bounded.
MAX_STEPS = 200000


@dataclass
class Scenario:
    """User-facing scenario settings (all SI units).

    Horizontal velocity `v0_horizontal_m_s` is applied along +X in the
    inertial frame (documented simplification).
    """
    start_height_m: float = DEFAULT_START_HEIGHT_M
    start_offset_xy: tuple = DEFAULT_START_OFFSET_XY
    target_xy: tuple = DEFAULT_TARGET_XY
    v0_vertical_m_s: float = DEFAULT_V0_VERTICAL_M_S
    v0_horizontal_m_s: float = DEFAULT_V0_HORIZONTAL_M_S
    dt: float = DEFAULT_DT_S
    max_time_s: float = DEFAULT_MAX_TIME_S
    descent_speed_m_s: float = DEFAULT_DESCENT_SPEED_M_S
    dry_mass_kg: float = DEFAULT_DRY_MASS_KG

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("Zeitschritt dt muss > 0 sein")
        if self.dt > 1.0:
            raise ValueError("Zeitschritt dt ist zu gross (max 1.0 s)")
        if self.start_height_m < 0.0:
            raise ValueError("Start-Hoehe darf nicht negativ sein")
        if self.max_time_s <= 0.0:
            raise ValueError("max_time_s muss > 0 sein")
        if self.dry_mass_kg <= 0.0:
            raise ValueError("Trockenmasse muss > 0 sein")
        if self.max_time_s / self.dt > MAX_STEPS:
            raise ValueError(
                f"zu viele Schritte ({int(self.max_time_s/self.dt)}); "
                "dt vergroessern")

    def initial_state(self) -> State:
        """Fresh initial State (the simulator mutates the state it gets)."""
        self.validate()
        ox, oy = float(self.start_offset_xy[0]), float(self.start_offset_xy[1])
        return State(
            pos=np.array([ox, oy, float(self.start_height_m)]),
            vel=np.array([float(self.v0_horizontal_m_s), 0.0,
                          float(self.v0_vertical_m_s)]),
        )

    def target(self) -> np.ndarray:
        return np.array([float(self.target_xy[0]), float(self.target_xy[1]), 0.0])


def run_scenario_with(scenario: Scenario, guidance) -> list:
    """Run one scenario with an explicit guidance/controller object.

    Used both for the ideal run (guidance = LandingController, same as
    run_scenario) and for observed runs (guidance = ObservedController).
    The physics stays identical; only the object that produces the thrust
    command differs.
    """
    scenario.validate()
    mass = MassModel(dry_mass_kg=scenario.dry_mass_kg, fuel_kg=0.0)
    cfg = SimConfig(tf=scenario.max_time_s, dt=scenario.dt,
                    stop_at_ground=True, ground_z=0.0)
    sim = Simulator(cfg, guidance=guidance)
    return sim.run(scenario.initial_state(), mass, target=scenario.target())


def run_scenario(scenario: Scenario) -> list:
    """Ideal reference run: controller sees the true state (no sensors)."""
    ctrl = LandingController(descent_speed_m_s=scenario.descent_speed_m_s)
    return run_scenario_with(scenario, ctrl)


# ---------------------------------------------------------------- rotation demo
ROTATION_DEMO_MOMENT_BODY = (2.0, 0.0, 0.0)  # N*m, body-frame (learning demo)
ROTATION_DEMO_I_DIAG = (1.0, 2.0, 3.0)       # kg*m^2, diagonal inertia


def run_rotation_demo(moment_body=ROTATION_DEMO_MOMENT_BODY,
                      tf: float = 2.0, dt: float = 0.02,
                      I_diag=ROTATION_DEMO_I_DIAG,
                      start_height_m: float = 5.0) -> list:
    """Pure rotation history (no translation dynamics, no gravity).

    Uses the existing rigid-body physics (RigidBody + integrate_rotation_body)
    through the Simulator's opt-in rotation flag. Kept separate from the landing
    scenario; does NOT touch controller or scenario defaults.
    """
    rb = RigidBody(mass_kg=DEFAULT_DRY_MASS_KG, I_diag=list(I_diag))
    state = State(
        pos=np.array([0.0, 0.0, float(start_height_m)]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_b=np.zeros(3),
        moment_total=np.asarray(moment_body, dtype=float),
    )
    cfg = SimConfig(tf=tf, dt=dt, gravity=False,
                    integrate_rotation_body=True, rigid_body=rb)
    sim = Simulator(cfg)
    mass = MassModel(dry_mass_kg=DEFAULT_DRY_MASS_KG, fuel_kg=0.0)
    return sim.run(state, mass)
