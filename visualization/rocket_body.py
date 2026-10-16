"""Simple 3D rocket geometry + transforms for the learning visualization.

This module is PRESENTATION ONLY. It builds a coarse polygonal rocket (nose,
cylindrical body) and a separate engine-nozzle bell, and transforms them with
the already existing quaternion rotation. There is NO physics/thrust/engine
model here.

Conventions (matching the rest of the project):
- Body frame: rocket long axis is +Z (base at z=0, tip at z=+length).
- q is scalar-first [w, x, y, z] = q_{B->I}; world vectors are q rotated by it.
- Nozzle hangs below the base (z <= 0) and is tilted about a hinge at the
  base center (0, 0, 0) by a demo gimbal angle.
"""
import numpy as np

from physics.rotation import rotate_vector


def _ring(radius, z, n):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang),
                            np.full(n, float(z))])


def rocket_geometry(length=1.0, radius=0.09, nose_fraction=0.35, n=16):
    """Return (vertices[N,3], faces[list of index lists]) in body frame.

    Closed body: base cap + cylindrical side + conical nose. Axis +Z.
    """
    z_cone = length * (1.0 - nose_fraction)
    base = _ring(radius, 0.0, n)
    top = _ring(radius, z_cone, n)
    tip = np.array([[0.0, 0.0, float(length)]])
    base_center = np.array([[0.0, 0.0, 0.0]])
    verts = np.vstack([base, top, tip, base_center])
    i_tip = 2 * n
    i_cen = 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])            # cylinder side
        faces.append([i_cen, j, i])                   # base cap
        faces.append([n + i, n + j, i_tip])           # nose cone
    return verts, faces


def nozzle_geometry(length=0.26, r_top=0.05, r_bottom=0.09, n=16):
    """Return (vertices[N,3], faces) of a downward flaring bell (body frame)."""
    top = _ring(r_top, 0.0, n)
    bottom = _ring(r_bottom, -abs(length), n)
    bottom_center = np.array([[0.0, 0.0, -abs(length)]])
    verts = np.vstack([top, bottom, bottom_center])
    i_cen = 2 * n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j, n + i])            # bell side
        faces.append([i_cen, n + j, n + i])           # bottom cap
    return verts, faces


def rotate_about_axis(points, axis, angle, origin=None):
    """Rodrigues rotation of points about an axis through origin."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    a = np.asarray(axis, dtype=float)
    a = a / (np.linalg.norm(a) + 1e-18)
    if origin is None:
        origin = np.zeros(3)
    origin = np.asarray(origin, dtype=float)
    p = pts - origin
    c, s = np.cos(angle), np.sin(angle)
    cross = np.cross(a, p)
    dot = p @ a
    rot = p * c + cross * s + np.outer(dot, a) * (1.0 - c)
    return rot + origin


def transform_points(points, q, position):
    """Rotate body-frame points by q and translate to world position."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    pos = np.asarray(position, dtype=float)
    return np.array([rotate_vector(q, p) for p in pts]) + pos


def body_axis_world(q):
    """World direction of the rocket long axis (base -> tip)."""
    return np.asarray(rotate_vector(q, np.array([0.0, 0.0, 1.0])), dtype=float)


def nozzle_axis_world(q, gimbal_angle=0.0, gimbal_axis=(0.0, 1.0, 0.0)):
    """World direction of the nozzle axis (points down/out of the bell)."""
    d_body = rotate_about_axis(np.array([[0.0, 0.0, -1.0]]),
                               gimbal_axis, gimbal_angle)[0]
    return np.asarray(rotate_vector(q, d_body), dtype=float)
