"""Unified 3D flight simulation (translation + rotation in ONE shared loop).

Learning model only. NOT a real vehicle/attitude controller.

Frame / convention contract (explicit):
- Inertial frame: ENU, z Up. World thrust and gravity vectors are inertial.
- Body frame: +Z along the rocket long axis (tip). q = [w,x,y,z] = q_{B->I}.
- All wrenches: the THRUST force is computed in the BODY frame from the
  gimbaled nozzle and rotated to inertial with q; gravity is inertial.
  The MOMENT is body-frame (omega_b/alpha_b/M_b) and drives rotation only.
- Forces affect translational state ONLY; moments affect rotational state ONLY.
  They are still linked through q (the same orientation rotates thrust and
  enters the equations of motion), which is the physically correct coupling.

Integration order (documented): 1) decide thrust+gimbal, 2) build wrench,
3) update translation (Euler-Cromer), 4) update rotation (semi-implicit Euler).
A quaternion is re-normalized after every step (`physics/rotation.py`).

The attitude/descent controller below is a SIMPLIFIED VIRTUAL MODEL (documented,
clearly not a real autopilot) that turns the gimbal into a real actuator so the
nozzle animation is a genuine physical input instead of an independent sine.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from config import units
from mass_model.mass import MassModel
from physics.rigidbody import RigidBody
from physics.rotation import rotate_vector, quat_inv, normalize
from physics.translation import Translation
from physics.thrust_model import ThrustModel
from simulation.loop import SimConfig, Simulator  # re-exported helpers
from state.core import State

__all__ = ["FlightConfig", "flight_telemetry", "run_flight",
           "SimConfig", "Simulator"]


@dataclass
class FlightConfig:
    t0: float = 0.0
    tf: float = 20.0
    dt: float = units.DEFAULT_DT
    gravity: bool = True
    stop_at_ground: bool = True
    ground_z: float = 0.0
    # initial state
    start_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 20.0]))
    start_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    start_omega_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # mass / inertia
    dry_mass_kg: float = 50.0
    fuel_kg: float = 10.0
    I_diag: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 3.0]))
    # simplified thrust MODEL
    max_thrust_N: float = 4000.0
    gimbal_max_rad: float = 0.35
    gimbal_offset_m: float = 0.30
    isp_s: float = 0.0        # >0 enables MODELLED fuel burn; 0 = constant mass
    # simplified model controller (virtual, not a real autopilot)
    use_controller: bool = True
    vz_des_m_s: float = -2.0
    kp_att: float = 4.0
    kd_att: float = 3.5
    kv_vz: float = 1.0
    # lateral (horizontal) velocity outer loop: desired horizontal velocity and
    # the gain that turns the horizontal-velocity error into a commanded tilt of
    # the thrust axis. kv_vh = 0 disables it (attitude hold on world +Z only).
    vh_des_m_s: np.ndarray = field(default_factory=lambda: np.zeros(2))
    kv_vh: float = 0.6
    max_tilt_deg: float = 20.0
    # open-loop inputs used when use_controller is False
    thrust_N_open: float = 0.0
    gimbal_x_open: float = 0.0
    gimbal_y_open: float = 0.0


def flight_telemetry(state: State, cfg: FlightConfig = None) -> dict:
    """Telemetry straight from ONE simulation state (single source of truth).

    Every requested quantity is either read from the state or explicitly
    flagged as NOT MODELLED. Nothing is invented for state fields.
    """
    m = float(state.mass_kg)
    g = units.GRAVITY_M_S2 if (cfg is None or cfg.gravity) else 0.0
    return {
        "time_s": float(state.t),
        "position_x_m": float(state.pos[0]),
        "position_y_m": float(state.pos[1]),
        "position_z_m": float(state.pos[2]),
        "height_m": float(state.height_m),
        "velocity_x_m_s": float(state.vel[0]),
        "velocity_y_m_s": float(state.vel[1]),
        "velocity_z_m_s": float(state.vel[2]),
        "speed_m_s": float(np.linalg.norm(state.vel)),
        "speed_horizontal_m_s": float(np.linalg.norm(np.asarray(state.vel[:2], float))),
        "acceleration_x_m_s2": float(state.acc[0]),
        "acceleration_y_m_s2": float(state.acc[1]),
        "acceleration_z_m_s2": float(state.acc[2]),
        "acceleration_mag_m_s2": float(np.linalg.norm(state.acc)),
        # orientation + rates
        "quaternion_B_to_I": np.asarray(state.q, dtype=float),
        "omega_b_rad_s": np.asarray(state.omega_b, dtype=float),
        "alpha_b_rad_s2": np.asarray(state.alpha_b, dtype=float),
        "omega_b_mag_rad_s": float(np.linalg.norm(state.omega_b)),
        "alpha_b_mag_rad_s2": float(np.linalg.norm(state.alpha_b)),
        # mass (present in this model)
        "dry_mass_kg": float(state.dry_mass_kg),
        "fuel_kg": float(state.fuel_kg),
        "total_mass_kg": m,
        "fuel_modeled": bool(state.modeled_thrust and m > 0 and state.fuel_kg >= 0
                             and (cfg is not None and cfg.isp_s > 0)),
        # forces (inertial frame), moment (body frame)
        "thrust_N": float(state.thrust_N),
        "gimbal_x_rad": float(state.gimbal_x),
        "gimbal_y_rad": float(state.gimbal_y),
        "force_thrust_N": np.asarray(state.force_thrust, dtype=float),
        "force_thrust_mag_N": float(np.linalg.norm(state.force_thrust)),
        "force_thrust_dir_world": (
            np.asarray(state.force_thrust) / max(np.linalg.norm(state.force_thrust), 1e-12)
            if np.linalg.norm(state.force_thrust) > 1e-12 else np.zeros(3)),
        "force_thrust_application_point_m": np.array([0.0, 0.0, 0.0]),
        "force_gravity_N": np.asarray(state.force_gravity, dtype=float),
        "force_gravity_mag_N": float(np.linalg.norm(state.force_gravity)),
        "force_total_N": np.asarray(state.force_total, dtype=float),
        "force_total_mag_N": float(np.linalg.norm(state.force_total)),
        "moment_total_body_Nm": np.asarray(state.moment_total, dtype=float),
        "moment_total_mag_Nm": float(np.linalg.norm(state.moment_total)),
        # modelled / not modelled flags
        "modeled_thrust": bool(state.modeled_thrust),
        "thrust_model_note": (
            "vereinfachtes virtuelles Schub-/Gimbal-Modell"
            if state.modeled_thrust else "Schub nicht modelliert"),
        "drag_modeled": False,
        "aerodynamics_modeled": False,
        "thrust_application_point_m": [0.0, 0.0, -getattr(cfg, "gimbal_offset_m", 0.0)]
        if cfg is not None else [0.0, 0.0, 0.0],
        "gravity_g0": float(g),
    }


def _model_control(state: State, m: float, thr: ThrustModel, cfg: FlightConfig):
    """Simplified VIRTUAL controller: descent speed + lateral speed + attitude.

    Three cascaded loops, all virtual (a learning model, not a real autopilot):
    1) vertical: thrust magnitude tracks a desired sink rate `vz_des`;
    2) lateral: the horizontal velocity error is turned into a desired TILT of
       the thrust axis; tilting the body is what actually produces the
       horizontal thrust component that opposes the drift, so this is the only
       way to remove an existing lateral velocity. With `kv_vh = 0` the desired
       tilt is zero (legacy behaviour);
    3) attitude: the TVC loop via the gimbal tracks the desired thrust axis.

    Returns (thrust_N, gimbal_x, gimbal_y). The gimbal angles feed the physics,
    so the nozzle motion is a genuine input to force AND moment.
    """
    g = units.GRAVITY_M_S2
    # --- vertical speed command -> thrust magnitude (inertial, along +Z when upright)
    thrust_N = m * (g + cfg.kv_vz * (cfg.vz_des_m_s - state.vel[2]))

    # --- lateral velocity -> desired tilt of the thrust axis (world frame) ---
    # thrust acceleration ~ thrust_N / m along the body axis; to cancel the
    # horizontal velocity the axis must lean against it. Small-angle:
    #   horizontal accel = (thrust/m) * tilt  =  -kv_vh * v_h
    thrust_acc = max(thrust_N / max(m, 1e-9), 1e-6)
    vh_err = np.asarray(state.vel[:2], float) - np.asarray(cfg.vh_des_m_s, float)
    max_tilt = np.deg2rad(cfg.max_tilt_deg)
    tilt = np.clip(-cfg.kv_vh * vh_err / thrust_acc, -max_tilt, max_tilt)
    u_des_world = normalize(np.array([tilt[0], tilt[1], 1.0]))

    # --- attitude hold onto the desired thrust direction (world) ---
    u_b = rotate_vector(quat_inv(state.q), u_des_world)
    e_b = np.array([u_b[1], -u_b[0], 0.0])          # cross(z_b, u_b) in body
    alpha_des = -cfg.kp_att * e_b - cfg.kd_att * np.asarray(state.omega_b, float)
    M_des = np.array([cfg.I_diag[0] * alpha_des[0],
                      cfg.I_diag[1] * alpha_des[1],
                      0.0])
    gx, gy = thr.gimbal_for_moment(M_des, thrust_N)
    thrust_N, gx, gy = thr.clamp(thrust_N, gx, gy)
    return thrust_N, gx, gy


def run_flight(cfg: FlightConfig = None) -> List[State]:
    """Run the unified flight: translation AND rotation share one loop.

    Returns a list of `State` snapshots; every snapshot carries the full
    telemetry (forces, moment, orientation, rates, mass) used by the view.
    """
    cfg = cfg or FlightConfig()
    if cfg.dt <= 0.0:
        raise ValueError("dt must be > 0")
    n_steps = int(round((cfg.tf - cfg.t0) / cfg.dt))
    if n_steps <= 0:
        return []

    rb = RigidBody(mass_kg=cfg.dry_mass_kg + cfg.fuel_kg, I_diag=list(cfg.I_diag))
    mass = MassModel(dry_mass_kg=cfg.dry_mass_kg, fuel_kg=cfg.fuel_kg)
    thr = ThrustModel(max_thrust_N=cfg.max_thrust_N,
                      gimbal_max_rad=cfg.gimbal_max_rad,
                      gimbal_offset_m=cfg.gimbal_offset_m)
    trans = Translation(cfg.dt)

    state = State(pos=np.asarray(cfg.start_pos, float).copy(),
                  vel=np.asarray(cfg.start_vel, float).copy(),
                  q=normalize(np.asarray(cfg.start_q, float)),
                  omega_b=np.asarray(cfg.start_omega_b, float).copy())
    history: List[State] = []

    for i in range(n_steps):
        state.t = cfg.t0 + i * cfg.dt
        state.sync_from_mass(mass)
        m = mass.total_mass()

        # 1) actuator input (controller or open-loop) -> simplified model
        if cfg.use_controller:
            thrust_N, gx, gy = _model_control(state, m, thr, cfg)
        else:
            thrust_N, gx, gy = (cfg.thrust_N_open, cfg.gimbal_x_open, cfg.gimbal_y_open)
        F_b, M_b, n_b = thr.force_moment(thrust_N, gx, gy)

        # 2) wrench: body thrust -> inertial; gravity inertial (mass dependent)
        F_i = rotate_vector(state.q, F_b)
        F_g = (m * np.array([0.0, 0.0, -units.GRAVITY_M_S2])
               if cfg.gravity else np.zeros(3))
        forces = F_i + F_g

        state.thrust_N = float(thrust_N)
        state.gimbal_x = float(gx)
        state.gimbal_y = float(gy)
        state.force_thrust = F_i.copy()
        state.force_gravity = F_g.copy()
        state.thrust_dir_b = n_b.copy()
        state.modeled_thrust = True
        state.moment_total = M_b.copy()

        # 3) translation (Euler-Cromer) -> affects pos/vel/acc ONLY
        trans.step(state, forces, m)

        # 4) rotation (semi-implicit) -> affects q/omega_b/alpha_b ONLY
        integrate_rotation_body = _body_rotator()
        integrate_rotation_body(state, cfg.dt, rb=rb)

        # optional MODELLED fuel burn
        if cfg.isp_s and cfg.isp_s > 0.0 and state.thrust_N > 0.0:
            mass.burn(state.thrust_N / (cfg.isp_s * units.GRAVITY_M_S2), cfg.dt)

        if cfg.stop_at_ground and state.pos[2] <= cfg.ground_z:
            state.pos = np.array([state.pos[0], state.pos[1], cfg.ground_z])
            state.height_m = cfg.ground_z
            history.append(state.copy())
            break

        history.append(state.copy())
    return history


def _body_rotator():
    from physics.rotation import integrate_rotation_body
    return integrate_rotation_body
