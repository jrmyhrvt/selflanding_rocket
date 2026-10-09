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
    # scalar-first
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def quat_inv(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z]) / (w*w + x*x + y*y + z*z + 1e-18)

def rotate_vector(q, v):
    vq = np.array([0.0, v[0], v[1], v[2]])
    res = quat_mul(quat_mul(q, vq), quat_inv(q))
    return res[1:]

def quat_to_euler(q):
    w, x, y, z = q
    # roll, pitch, yaw
    sinr_cosp = 2.0*(w*x + y*z)
    cosr_cosp = 1.0 - 2.0*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])
