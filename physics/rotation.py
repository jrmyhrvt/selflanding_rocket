"""Quaternion helpers and simple rotation utilities."""
import numpy as np


def norm(q):
    return np.linalg.norm(q)


def normalize(q):
    n = norm(q)
    if n < 1e-18:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (w * w + x * x + y * y + z * z + 1e-18)


def rotate_vector(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]])
    res = quat_mul(quat_mul(q, vq), quat_inv(q))
    return res[1:]


def quat_to_euler(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


def integrate_rotation(state, dt):
    """
    Legacy rotation integration (simplified). Uses inertial-frame interpretation
    for omega/alpha as originally implemented. Not used by body-frame path.
    """
    if dt <= 0.0:
        return state
    state.omega = np.asarray(state.omega, dtype=float) + np.asarray(state.alpha, dtype=float) * dt
    omega = np.asarray(state.omega, dtype=float)
    w, x, y, z = np.asarray(state.q, dtype=float)
    ow, ox, oy, oz = 0.0, omega[0], omega[1], omega[2]
    q_dot_w = 0.5 * (w * ow - x * ox - y * oy - z * oz)
    q_dot_x = 0.5 * (w * ox + x * ow + y * oz - z * oy)
    q_dot_y = 0.5 * (w * oy - x * oz + y * ow + z * ox)
    q_dot_z = 0.5 * (w * oz + x * oy - y * ox + z * ow)
    q_new = np.array([w + q_dot_w * dt, x + q_dot_x * dt, y + q_dot_y * dt, z + q_dot_z * dt])
    state.q = normalize(q_new)
    state.omega = omega
    return state


def integrate_rotation_body(state, dt, rb=None):
    """
    Body-frame rotation integration (q_{B->I}, scalar-first).
    - omega_b, alpha_b, M_b in body frame
    - q_dot = 0.5*q ⊗ (0, omega_b)
    - if rb: alpha_b = rb.angular_accel(omega_b, M_b), M_b = state.moment_total
    """
    if dt <= 0.0:
        return state
    omega_b = np.asarray(state.omega_b, dtype=float)
    M_b = np.asarray(state.moment_total, dtype=float)
    if rb is not None:
        alpha_b = rb.angular_accel(omega_b, M_b)
    else:
        alpha_b = np.asarray(state.alpha_b, dtype=float)
    state.alpha_b = alpha_b.copy()
    omega_b_new = omega_b + alpha_b * dt
    w, x, y, z = np.asarray(state.q, dtype=float)
    ow, ox, oy, oz = 0.0, omega_b_new[0], omega_b_new[1], omega_b_new[2]
    q_dot_w = 0.5 * (w * ow - x * ox - y * oy - z * oz)
    q_dot_x = 0.5 * (w * ox + x * ow + y * oz - z * oy)
    q_dot_y = 0.5 * (w * oy - x * oz + y * ow + z * ox)
    q_dot_z = 0.5 * (w * oz + x * oy - y * ox + z * ow)
    q_new = np.array([w + q_dot_w * dt, x + q_dot_x * dt, y + q_dot_y * dt, z + q_dot_z * dt])
    state.q = normalize(q_new)
    state.omega_b = omega_b_new
    return state
