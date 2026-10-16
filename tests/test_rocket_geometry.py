"""Tests for the presentation-only 3D rocket geometry."""
import numpy as np
import pytest

from visualization.rocket_body import (
    rocket_geometry, nozzle_geometry, transform_points, rotate_about_axis,
    body_axis_world, nozzle_axis_world)


def test_rocket_geometry_vertices_and_tip():
    verts, faces = rocket_geometry(length=1.0, radius=0.09, nose_fraction=0.35, n=16)
    assert verts.shape == (34, 3)          # 16 base + 16 top + tip + centre
    assert len(faces) > 0
    assert np.isclose(verts[:, 2].max(), 1.0)   # tip at +length
    assert np.isclose(verts[:, 2].min(), 0.0)   # base at 0
    # all face indices valid
    for f in faces:
        assert min(f) >= 0 and max(f) < len(verts)


def test_nozzle_geometry_hangs_below_base():
    verts, faces = nozzle_geometry(length=0.26, r_top=0.05, r_bottom=0.09, n=16)
    assert verts.shape == (33, 3)          # 16 top + 16 bottom + centre
    assert np.isclose(verts[:, 2].min(), -0.26)
    assert np.isclose(verts[:, 2].max(), 0.0)
    assert verts[:, 2].max() > verts[:, 2].min()
    for f in faces:
        assert min(f) >= 0 and max(f) < len(verts)


def test_nozzle_bell_flares_outwards():
    verts, _ = nozzle_geometry(length=0.26, r_top=0.05, r_bottom=0.09, n=16)
    r_top = np.hypot(verts[:16, 0], verts[:16, 1])
    r_bot = np.hypot(verts[16:32, 0], verts[16:32, 1])
    assert np.allclose(r_top, 0.05)
    assert np.allclose(r_bot, 0.09)


def test_transform_points_applies_quaternion_and_translation():
    q90x = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    pts = np.array([[0.0, 0.0, 1.0]])
    out = transform_points(pts, q90x, np.array([5.0, 0.0, 0.0]))
    assert np.allclose(out[0], [5.0, -1.0, 0.0], atol=1e-9)


def test_rotate_about_axis_identity_when_zero_angle():
    p = np.array([[0.0, 0.0, -1.0]])
    assert np.allclose(rotate_about_axis(p, (0, 1, 0), 0.0), p)


def test_rotate_about_axis_tilts_nozzle():
    p = np.array([[0.0, 0.0, -1.0]])
    out = rotate_about_axis(p, (0, 1, 0), np.pi / 2.0)[0]
    assert np.allclose(out, [-1.0, 0.0, 0.0], atol=1e-9)


def test_body_axis_world_identity_and_rotation():
    assert np.allclose(body_axis_world([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 1.0])
    q90x = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    assert np.allclose(body_axis_world(q90x), [0.0, -1.0, 0.0], atol=1e-9)


def test_nozzle_axis_world_with_gimbal_angle():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(nozzle_axis_world(q, 0.0), [0.0, 0.0, -1.0], atol=1e-9)
    theta = np.deg2rad(30.0)
    nax = nozzle_axis_world(q, theta)
    cosang = float(np.clip(np.dot(nax, -np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
    assert np.isclose(np.arccos(cosang), theta, atol=1e-9)
