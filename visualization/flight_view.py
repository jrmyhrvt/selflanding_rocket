"""Unified 3D flight view: ONE run, rocket + movable nozzle, force vectors,
live telemetry and plots.

Presentation only: every displayed number comes from the same simulation state
used for the animation (`simulation/flight.py`), read through
`flight_telemetry()`. Nothing is animated independently of the physics, and
quantities the model does not compute are explicitly labelled as not modelled.

Visualization notes (learning simulation):
- The rocket hull and the movable nozzle are rebuilt from the CURRENT state on
  every frame (position, orientation `q`, gimbal angles) - never a fake overlay.
- The body long axis (+Z) and the ACTUAL thrust direction (+n_b rotated by q)
  are drawn as separate lines so the gimbal tilt is visible; they only coincide
  when the gimbal is zero.
- Force/velocity/acceleration arrows are scaled by their own physical magnitude
  (normalized to the running maximum of the same quantity), NOT by the rocket
  geometry, so they stay a fixed on-screen size and never swamp the hull.
- The camera (azimuth/elevation/zoom/pan) is owned by this view and is NOT reset
  per frame; only an explicit view-mode switch or "Ansicht reset" reframes it.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from config import units
from physics.rotation import quat_to_euler, rotate_vector
from simulation.flight import FlightConfig, flight_telemetry, run_flight
from visualization.rocket_body import (rocket_geometry, nozzle_geometry,
                                       transform_points, rotate_about_axis)

MODE_UPRIGHT = "Aufrecht"
MODE_TILTED = "Geneigt (15 Grad)"
VIEW_OVERVIEW = "Gesamtansicht"
VIEW_CLOSEUP = "Nahansicht"
CTRL_AUTO = "Auto (Regler)"
CTRL_MANUAL = "Manuell (Servo)"
ROCKET_LENGTH_M = 1.0
NOZZLE_OFFSET_M = 0.30


def _fig_text_axis(fig, x0, y0, w, h, title):
    ax = fig.add_axes([x0, y0, w, h])
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("0.7")
    return ax


@dataclass(eq=False)
class FlightView:
    """Interactive, single-state-source flight visualization."""

    cfg: FlightConfig = None
    history: list = None
    paused: bool = False

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = FlightConfig()
        if self.history is None:
            self.history = run_flight(self.cfg)
        self.frame = 0
        self.ani = None
        # --- camera / view state (owned by the view, never reset per frame) ---
        self._viz_mode = VIEW_OVERVIEW
        self._cam_center = np.zeros(3)
        self._cam_half = 10.0
        self._cam_elev = 22.0
        self._cam_azim = -60.0
        self._pan_offset = np.zeros(3)
        self._drag = None
        self._arrow_len = 1.0
        self._status_note = ""
        self._build_figure()
        self._cache_arrays()
        self._reset_axes()
        self.update(0)

    # ------------------------------------------------------------------ build
    def _build_figure(self):
        self.fig = plt.figure(figsize=(15, 9))
        gs = self.fig.add_gridspec(3, 4, wspace=0.35, hspace=0.5,
                                   left=0.05, right=0.98, top=0.93, bottom=0.215)

        self.ax3d = self.fig.add_subplot(gs[0:2, 0:2], projection="3d")
        self.ax3d.set_title("3D-Ansicht (EIN Simulationslauf)", fontsize=10)
        self.trail, = self.ax3d.plot([], [], [], "k-", alpha=0.35, lw=1.0)
        self.body_poly = Poly3DCollection([], facecolor="0.8",
                                          edgecolor="0.3", alpha=0.9)
        self.nozzle_poly = Poly3DCollection([], facecolor="0.4",
                                            edgecolor="0.1", alpha=0.95)
        self.ax3d.add_collection3d(self.body_poly)
        self.ax3d.add_collection3d(self.nozzle_poly)
        self._vec_arrows = []
        self._vec_labels = []
        self._axis_lines = []
        self._axes_arrows = []

        self.ax_tele = _fig_text_axis(self.fig, 0.50, 0.40, 0.235, 0.51,
                                      "Live-Telemetrie")
        self.tele_text = self.ax_tele.text(0.02, 0.98, "", family="monospace",
                                           fontsize=7.5, va="top",
                                           transform=self.ax_tele.transAxes)

        self.ax_h = self.fig.add_subplot(gs[0, 3])
        self.ax_v = self.fig.add_subplot(gs[1, 3])
        self.ax_r = self.fig.add_subplot(gs[2, :])
        for ax, title, ylab in (
                (self.ax_h, "Hoehe ueber Zeit", "h [m]"),
                (self.ax_v, "Betrag / Komponenten [m/s, m/s^2]", "vel/acc"),
                (self.ax_r, "Drehung ueber Zeit (Aktuator: Gimbal)", "w[rad/s], Gimbal[deg]")):
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("t [s]", fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        # ---- row 1 (bottom): simulation + camera controls
        ax_start = self.fig.add_axes([0.05, 0.02, 0.075, 0.045])
        ax_pause = self.fig.add_axes([0.135, 0.02, 0.075, 0.045])
        ax_restart = self.fig.add_axes([0.22, 0.02, 0.085, 0.045])
        ax_dt = self.fig.add_axes([0.315, 0.02, 0.06, 0.045])
        ax_mode = self.fig.add_axes([0.39, 0.012, 0.115, 0.065])
        ax_view = self.fig.add_axes([0.515, 0.012, 0.115, 0.065])
        ax_zin = self.fig.add_axes([0.645, 0.05, 0.05, 0.028])
        ax_zout = self.fig.add_axes([0.645, 0.016, 0.05, 0.028])
        ax_vreset = self.fig.add_axes([0.705, 0.016, 0.095, 0.062])
        self.b_start = Button(ax_start, "Start")
        self.b_pause = Button(ax_pause, "Pause")
        self.b_restart = Button(ax_restart, "Neustart")
        self.box_dt = TextBox(ax_dt, "dt=", initial=str(self.cfg.dt))
        self.radios = RadioButtons(ax_mode, (MODE_UPRIGHT, MODE_TILTED))
        self.radios.set_active(0)
        self.radios_view = RadioButtons(ax_view, (VIEW_OVERVIEW, VIEW_CLOSEUP))
        self.radios_view.set_active(0)
        self.b_zin = Button(ax_zin, "Zoom +")
        self.b_zout = Button(ax_zout, "Zoom -")
        self.b_vreset = Button(ax_vreset, "Ansicht reset")

        # ---- row 2 (top): control mode (auto/manual) + the two servo test sliders
        gmax_deg = float(np.degrees(self.cfg.gimbal_max_rad))
        ax_ctrl = self.fig.add_axes([0.05, 0.105, 0.115, 0.075])
        ax_sx = self.fig.add_axes([0.20, 0.158, 0.26, 0.026])
        ax_sy = self.fig.add_axes([0.20, 0.114, 0.26, 0.026])
        self.ax_cstatus = _fig_text_axis(self.fig, 0.49, 0.10, 0.49, 0.085,
                                         "Steuerung / Servos")
        self.cstatus_text = self.ax_cstatus.text(
            0.01, 0.95, "", family="monospace", fontsize=8, va="top",
            transform=self.ax_cstatus.transAxes)
        self.radios_ctrl = RadioButtons(ax_ctrl, (CTRL_AUTO, CTRL_MANUAL))
        self.radios_ctrl.set_active(0)
        self.slider_sx = Slider(ax_sx, "Servo X [deg]", -gmax_deg, gmax_deg,
                                valinit=float(np.degrees(self.cfg.servo_x0_rad)),
                                valstep=0.1)
        self.slider_sy = Slider(ax_sy, "Servo Y [deg]", -gmax_deg, gmax_deg,
                                valinit=float(np.degrees(self.cfg.servo_y0_rad)),
                                valstep=0.1)

        self.b_start.on_clicked(lambda e: self.play())
        self.b_pause.on_clicked(lambda e: self.toggle_pause())
        self.b_restart.on_clicked(lambda e: self.restart())
        self.box_dt.on_submit(lambda text: self.set_dt(text))
        self.radios.on_clicked(self._on_mode)
        self.radios_view.on_clicked(self._on_view_mode)
        self.radios_ctrl.on_clicked(self._on_ctrl_mode)
        self.slider_sx.on_changed(self._on_servo_x)
        self.slider_sy.on_changed(self._on_servo_y)
        self.b_zin.on_clicked(lambda e: self.zoom(0.8))
        self.b_zout.on_clicked(lambda e: self.zoom(1.25))
        self.b_vreset.on_clicked(lambda e: self.reset_view())

        self._build_series()
        self._install_interaction()

    def _install_interaction(self):
        """Own the 3D camera: rotate (left drag), pan (right/middle drag),
        zoom (wheel / keys / buttons). Built-in mouse rotation is disabled so a
        user rotation is never silently overwritten by a per-frame reframe."""
        canvas = self.fig.canvas
        self.ax3d.disable_mouse_rotation()
        self._cids = [
            canvas.mpl_connect("scroll_event", self._on_scroll),
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
            canvas.mpl_connect("key_press_event", self._on_key),
        ]

    def _build_series(self):
        self.hline, = self.ax_h.plot([], [], "g-", alpha=0.7, lw=1.1)
        self.vline_mag, = self.ax_v.plot([], [], "b-", alpha=0.8, lw=1.1, label="|v|")
        self.vline_z, = self.ax_v.plot([], [], "m-", alpha=0.7, lw=1.0, label="v_z")
        self.vline_h, = self.ax_v.plot([], [], "c-", alpha=0.9, lw=1.1, label="|v_h|")
        self.aline_mag, = self.ax_v.plot([], [], "r--", alpha=0.7, lw=1.0, label="|a|")
        self.ax_v.legend(loc="upper right", fontsize=7)
        self.oline, = self.ax_r.plot([], [], "b-", alpha=0.8, lw=1.1, label="|omega_b|")
        self.gline_x, = self.ax_r.plot([], [], "g-", alpha=0.8, lw=1.1, label="Servo X Ist [deg/10]")
        self.gline_y, = self.ax_r.plot([], [], "c-", alpha=0.8, lw=1.1, label="Servo Y Ist [deg/10]")
        self.gline_x_cmd, = self.ax_r.plot([], [], "g--", alpha=0.5, lw=1.0, label="Servo X Soll")
        self.gline_y_cmd, = self.ax_r.plot([], [], "c--", alpha=0.5, lw=1.0, label="Servo Y Soll")
        self.ax_r.legend(loc="upper right", fontsize=6, ncol=2)
        self.hmark, = self.ax_h.plot([], [], "go", ms=5)
        self.vmark, = self.ax_v.plot([], [], "bo", ms=5)
        self.rmark, = self.ax_r.plot([], [], "bo", ms=5)

    # ------------------------------------------------------------------ arrays
    def _cache_arrays(self):
        h = self.history
        if not h:
            self._t = np.array([])
            for name in ("_x", "_y", "_z", "_gim_x", "_gim_y",
                         "_gim_x_cmd", "_gim_y_cmd"):
                setattr(self, name, np.array([]))
            for name in ("_q", "_vel", "_acc", "_omega_b", "_alpha_b"):
                setattr(self, name, np.zeros((0, 3)))
            self._cache_forces = []
            self._ref = {"v": 1.0, "a": 1.0, "Ft": 1.0, "Fg": 1.0, "Ftot": 1.0}
            return
        self._t = np.array([s.t for s in h])
        self._x = np.array([s.pos[0] for s in h])
        self._y = np.array([s.pos[1] for s in h])
        self._z = np.array([s.height_m for s in h])
        self._q = np.array([np.asarray(s.q, dtype=float) for s in h])
        self._vel = np.array([np.asarray(s.vel, dtype=float) for s in h])
        self._acc = np.array([np.asarray(s.acc, dtype=float) for s in h])
        self._omega_b = np.array([np.asarray(s.omega_b, dtype=float) for s in h])
        self._alpha_b = np.array([np.asarray(s.alpha_b, dtype=float) for s in h])
        self._gim_x = np.array([s.gimbal_x for s in h])
        self._gim_y = np.array([s.gimbal_y for s in h])
        self._gim_x_cmd = np.array([s.gimbal_x_cmd for s in h])
        self._gim_y_cmd = np.array([s.gimbal_y_cmd for s in h])
        self._cache_forces = [(np.asarray(s.force_thrust, float),
                               np.asarray(s.force_gravity, float),
                               np.asarray(s.force_total, float)) for s in h]

        def _refmax(arr):
            if not len(arr):
                return 1.0
            return float(max(np.linalg.norm(arr, axis=1).max(), 1e-9))

        ft = np.array([f[0] for f in self._cache_forces])
        fg = np.array([f[1] for f in self._cache_forces])
        ftot = np.array([f[2] for f in self._cache_forces])
        self._ref = {"v": _refmax(self._vel), "a": _refmax(self._acc),
                     "Ft": _refmax(ft), "Fg": _refmax(fg), "Ftot": _refmax(ftot)}

    def _series_limits(self):
        if not len(self._t):
            return
        tmax = max(self._t.max(), 1e-6)
        for ax in (self.ax_h, self.ax_v, self.ax_r):
            ax.set_xlim(0.0, tmax)
        self.ax_h.set_ylim(0.0, max(1.0, self._z.max() * 1.05))
        vmax = max(1.0, float(np.abs(self._vel).max()) * 1.1,
                   float(np.abs(self._acc).max()) * 1.1)
        self.ax_v.set_ylim(-vmax, vmax)
        omax = max(1e-3, float(np.abs(self._omega_b).max()) * 1.3,
                   float(np.abs(self._gim_x).max() * 180.0 / np.pi / 10.0),
                   float(np.abs(self._gim_y).max() * 180.0 / np.pi / 10.0),
                   float(np.abs(self._gim_x_cmd).max() * 180.0 / np.pi / 10.0),
                   float(np.abs(self._gim_y_cmd).max() * 180.0 / np.pi / 10.0))
        self.ax_r.set_ylim(-omax, omax)
        self.hline.set_data(self._t, self._z)
        self.vline_mag.set_data(self._t, np.linalg.norm(self._vel, axis=1))
        self.vline_z.set_data(self._t, self._vel[:, 2])
        self.vline_h.set_data(self._t, np.linalg.norm(self._vel[:, :2], axis=1))
        self.aline_mag.set_data(self._t, np.linalg.norm(self._acc, axis=1))
        self.oline.set_data(self._t, np.linalg.norm(self._omega_b, axis=1))
        self.gline_x.set_data(self._t, np.degrees(self._gim_x) / 10.0)
        self.gline_y.set_data(self._t, np.degrees(self._gim_y) / 10.0)
        self.gline_x_cmd.set_data(self._t, np.degrees(self._gim_x_cmd) / 10.0)
        self.gline_y_cmd.set_data(self._t, np.degrees(self._gim_y_cmd) / 10.0)

    def _reset_axes(self):
        if not len(self._t):
            return
        self._series_limits()
        if self._viz_mode == VIEW_CLOSEUP:
            self._frame_closeup(self.frame)
        else:
            self._frame_overview()
        self._apply_camera()
        self._draw_world()

    # ------------------------------------------------------------------ camera
    def _frame_overview(self):
        cx, cy, cz = (float(self._x.mean()), float(self._y.mean()),
                      float(self._z.mean()))
        span = max(float(np.ptp(self._x)), float(np.ptp(self._y)),
                   float(np.ptp(self._z)), 1.0)
        self._cam_center = np.array([cx, cy, cz])
        self._cam_half = 0.62 * span
        self._pan_offset = np.zeros(3)

    def _frame_closeup(self, frame):
        self._cam_center = self._rocket_pos(frame) + np.array([0.0, 0.0, 0.35])
        self._cam_half = 1.0
        self._pan_offset = np.zeros(3)

    def _apply_camera(self):
        c = np.asarray(self._cam_center, float)
        h = max(float(self._cam_half), 1e-3)
        ax = self.ax3d
        ax.set_xlim(c[0] - h, c[0] + h)
        ax.set_ylim(c[1] - h, c[1] + h)
        zmin = c[2] - h
        if self._viz_mode == VIEW_OVERVIEW:
            zmin = min(zmin, 0.0)      # keep the ground plane (z=0) in view
        ax.set_zlim(zmin, c[2] + h)
        try:
            ax.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass
        ax.view_init(elev=float(np.clip(self._cam_elev, -89.0, 89.0)),
                     azim=float(self._cam_azim))

    def _draw_world(self):
        c, h = np.asarray(self._cam_center, float), float(self._cam_half)
        x0, x1 = c[0] - h, c[0] + h
        y0, y1 = c[1] - h, c[1] + h
        for a in self._axes_arrows:
            try:
                a.remove()
            except Exception:
                pass
        self._axes_arrows = []
        try:
            self._ground.remove()
        except Exception:
            pass
        xx, yy = np.meshgrid([x0, x1], [y0, y1])
        self._ground = self.ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                                              alpha=0.12, color="green")
        o = np.array([x0, y0, 0.0])
        for k, color in enumerate(("r", "b", "g")):
            v = np.zeros(3)
            v[k] = h * 0.5
            self._axes_arrows.append(
                self.ax3d.quiver(o[0], o[1], o[2], v[0], v[1], v[2],
                                 color=color, linewidths=1.4, normalize=False))
        self._axes_arrows.append(self.ax3d.text(o[0], o[1], o[2] + h * 0.5,
                                                "Welt x/y/z", fontsize=7, color="k"))

    def rotate(self, dazim, delev):
        """Rotate the camera (azimuth/elevation) and re-apply (no reframe)."""
        self._cam_azim += float(dazim)
        self._cam_elev = float(np.clip(self._cam_elev + float(delev), -89.0, 89.0))
        self._apply_camera()

    def zoom(self, factor):
        """Zoom the camera: factor < 1 zooms in, > 1 zooms out."""
        self._cam_half = float(np.clip(self._cam_half * float(factor), 0.15, 1.0e5))
        self._apply_camera()

    def pan(self, dx, dy):
        """Shift the camera centre (screen pixels -> world units)."""
        scale = self._cam_half / 500.0
        d = np.array([-float(dx) * scale, 0.0, float(dy) * scale])
        self._cam_center = np.asarray(self._cam_center, float) + d
        if self._viz_mode == VIEW_CLOSEUP:
            self._pan_offset = self._pan_offset + d
        self._apply_camera()

    def reset_view(self):
        """Reframe the current mode and restore the default orientation."""
        if self._viz_mode == VIEW_CLOSEUP:
            self._frame_closeup(self.frame)
        else:
            self._frame_overview()
        self._cam_elev, self._cam_azim = 22.0, -60.0
        self._apply_camera()

    def set_view_mode(self, label):
        """Switch between the whole-trajectory view and the rocket close-up.

        Reframes ONCE here; per-frame updates never reframe (a close-up only
        re-centres on the moving rocket, keeping the user's rotation/zoom).
        """
        if label == VIEW_CLOSEUP:
            self._viz_mode = VIEW_CLOSEUP
            self._frame_closeup(self.frame)
        else:
            self._viz_mode = VIEW_OVERVIEW
            self._frame_overview()
        self._cam_elev, self._cam_azim = 22.0, -60.0
        self._apply_camera()
        self.update(self.frame)

    # ------------------------------------------------------------- geometry/3d
    def body_world(self):
        return self._body_world

    def nozzle_world(self):
        return self._nozzle_world

    def _rocket_pos(self, frame):
        frame = int(min(max(frame, 0), len(self.history) - 1)) if self.history else 0
        return np.array([self._x[frame], self._y[frame], self._z[frame]])

    def _gimbal_angles(self, frame):
        return float(self._gim_x[frame]), float(self._gim_y[frame])

    def _nozzle_axis_body(self, gx, gy):
        """Body-frame nozzle axis (points down/out), from the SAME gimbal the
        physics uses. Equals -thrust_direction_body(gx, gy)."""
        d = rotate_about_axis(np.array([[0.0, 0.0, -1.0]]), [1.0, 0.0, 0.0], gx)
        d = rotate_about_axis(d, [0.0, 1.0, 0.0], gy)
        return d[0]

    def _rocket_world(self, frame):
        q = self._q[frame]
        pos = self._rocket_pos(frame)
        bv, bf = rocket_geometry(length=ROCKET_LENGTH_M)
        nv, nf = nozzle_geometry()
        gx, gy = self._gimbal_angles(frame)
        # the nozzle is hinged at the base centre and rotated by the SAME gimbal
        # that produces the thrust force/moment in the physics.
        nv = rotate_about_axis(nv, [1.0, 0.0, 0.0], gx)
        nv = rotate_about_axis(nv, [0.0, 1.0, 0.0], gy)
        return (transform_points(bv, q, pos), bf,
                transform_points(nv, q, pos), nf, q, pos)

    def _draw_rocket(self, frame):
        bw, bf, nw, nf, q, pos = self._rocket_world(frame)
        self._body_world, self._nozzle_world = bw, nw
        self.body_poly.set_verts([bw[fi] for fi in bf])
        self.nozzle_poly.set_verts([nw[fi] for fi in nf])

        self._clear_vectors()
        self._arrow_len = max(0.35 * float(self._cam_half), 1e-3)

        # body axes + long axis (green +Z = rocket long axis)
        axis_len = 0.6
        self._add_axes(pos, q, L=axis_len)

        # ---- actual thrust direction vs. rocket long axis (gimbal visible) ----
        st = self.history[frame]
        n_b = np.asarray(getattr(st, "thrust_dir_b", np.array([0.0, 0.0, 1.0])),
                         float)
        if not np.all(np.isfinite(n_b)) or np.allclose(n_b, 0.0):
            n_b = np.array([0.0, 0.0, 1.0])
        n_w = np.asarray(rotate_vector(q, n_b), float).ravel()
        body_dir = np.asarray(rotate_vector(q, np.array([0.0, 0.0, 1.0])),
                              float).ravel()
        p = np.asarray(pos, float)
        self._add_axis_line(p, p + n_w * axis_len, "darkorange", "Schubrichtung")
        self._add_axis_line(p, p - n_w * (axis_len * 0.55), "0.5",
                            "Duesenachse", ls="--")
        angle = float(np.degrees(np.arccos(np.clip(float(np.dot(body_dir, n_w)),
                                                   -1.0, 1.0))))
        gx, gy = self._gimbal_angles(frame)
        tip = p + body_dir * (ROCKET_LENGTH_M * 1.05)
        self._vec_labels.append(self.ax3d.text(
            tip[0], tip[1], tip[2],
            f"Gimbal x/y = {np.degrees(gx):+.1f}/{np.degrees(gy):+.1f} deg\n"
            f"Laengsachse vs. Schub = {angle:.1f} deg",
            fontsize=7.5, color="k"))

        # ---- the two servo rotation axes at the engine joint (hinge) ----------
        hinge = p + np.asarray(rotate_vector(
            q, np.array([0.0, 0.0, -NOZZLE_OFFSET_M])), float).ravel()
        self._add_servo_axes(hinge, q)

        # ---- labelled vectors (scaled by their own magnitude, not by the hull)
        self._add_vector(p, self._vel[frame], "cyan", "v", ref=self._ref["v"])
        self._add_vector(p, self._acc[frame], "magenta", "a", ref=self._ref["a"])
        f_thrust, f_grav, f_tot = self._cache_forces[frame]
        self._add_vector(p, f_thrust, "orange", "F_Schub", ref=self._ref["Ft"])
        self._add_vector(p, f_grav, "gray", "F_Schwerkraft", ref=self._ref["Fg"])
        self._add_vector(p, f_tot, "red", "F_gesamt", ref=self._ref["Ftot"])

    def _clear_vectors(self):
        for group in ("_vec_arrows", "_axis_lines"):
            for a in getattr(self, group, []):
                try:
                    a.remove()
                except Exception:
                    pass
            setattr(self, group, [])
        for t in getattr(self, "_vec_labels", []):
            try:
                t.remove()
            except Exception:
                pass
        self._vec_labels = []

    def _add_axes(self, origin, q, L=0.5):
        """Draw the rocket body axes (rotated by q) at the COM.

        +X red, +Y blue, +Z green (green = rocket long axis).
        """
        colors = ("r", "b", "g")
        names = ("Koerper-X", "Koerper-Y", "Koerper-Z (Laengsachse)")
        for k in range(3):
            e = np.zeros(3)
            e[k] = L
            v = rotate_vector(q, e)
            self._vec_arrows.append(
                self.ax3d.quiver(origin[0], origin[1], origin[2],
                                 v[0], v[1], v[2], color=colors[k],
                                 linewidths=1.2, normalize=False))
            tip = np.asarray(origin, float) + v
            self._vec_labels.append(
                self.ax3d.text(tip[0], tip[1], tip[2], names[k], fontsize=7,
                               color=colors[k]))

    def _add_servo_axes(self, hinge, q, L=0.45):
        """Draw the two servo ROTATION axes at the engine joint.

        Servo X turns the nozzle about body +X, servo Y about body +Y; both
        rotation axes are drawn (rotated by the rocket orientation q) so the
        servo geometry is visible at the hinge, independent of the nozzle tilt.
        """
        hinge = np.asarray(hinge, float)
        for axis_vec, color, name in (((1.0, 0.0, 0.0), "r", "Servo X (Achse)"),
                                      ((0.0, 1.0, 0.0), "b", "Servo Y (Achse)")):
            v = np.asarray(rotate_vector(q, np.asarray(axis_vec, float)),
                           float).ravel() * L
            self._vec_arrows.append(
                self.ax3d.quiver(hinge[0], hinge[1], hinge[2],
                                 v[0], v[1], v[2], color=color,
                                 linewidths=1.0, normalize=False, alpha=0.75))
            tip = hinge + v
            self._vec_labels.append(self.ax3d.text(
                tip[0], tip[1], tip[2], name, fontsize=6.5, color=color))

    def _add_axis_line(self, origin, end, color, label, ls="-"):
        o = np.asarray(origin, float)
        e = np.asarray(end, float)
        ln, = self.ax3d.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]],
                             color=color, lw=2.0, ls=ls)
        self._axis_lines.append(ln)
        self._vec_labels.append(self.ax3d.text(e[0], e[1], e[2], label,
                                               color=color, fontsize=8))

    def _vec_display_length(self, vec, ref=None):
        """World length used to draw `vec`: normalized to `ref` (running max of
        the same quantity) and capped at the view's arrow length. Independent of
        the rocket geometry."""
        mag = float(np.linalg.norm(np.asarray(vec, float)))
        if mag < 1e-12:
            return 0.0
        if ref is None or ref <= 0.0:
            return mag
        length = self._arrow_len * min(mag / ref, 1.0)
        return max(length, 0.12 * self._arrow_len)

    def _add_vector(self, origin, vec, color, label, ref=None):
        v = np.asarray(vec, dtype=float)
        mag = float(np.linalg.norm(v))
        if mag < 1e-12 or not np.all(np.isfinite(v)):
            return
        length = self._vec_display_length(v, ref)
        d = v / mag * length
        arrow = self.ax3d.quiver(origin[0], origin[1], origin[2],
                                 d[0], d[1], d[2], color=color,
                                 linewidths=1.6, normalize=False, alpha=0.9)
        self._vec_arrows.append(arrow)
        tip = np.asarray(origin, dtype=float) + d
        self._vec_labels.append(
            self.ax3d.text(tip[0], tip[1], tip[2], label,
                           color=color, fontsize=8, ha="left"))

    # ------------------------------------------------------------------ update
    def update(self, frame):
        if not self.history:
            return ()
        frame = int(min(max(frame, 0), len(self.history) - 1))
        self.frame = frame
        self._cur_frame = frame
        self._draw_rocket(frame)
        self.trail.set_data(self._x[:frame + 1], self._y[:frame + 1])
        self.trail.set_3d_properties(self._z[:frame + 1])
        t = self._t[frame]
        self.hmark.set_data([t], [self._z[frame]])
        self.vmark.set_data([t], [self._vel[frame, 2]])
        self.rmark.set_data([t], [np.linalg.norm(self._omega_b[frame])])
        self.tele_text.set_text(self._telemetry_text(frame))
        self.cstatus_text.set_text(self._control_status_text(frame))
        # In the close-up the camera follows the rocket, but it keeps the user's
        # rotation/zoom (only the centre moves) - it never "jumps back".
        if self._viz_mode == VIEW_CLOSEUP:
            self._cam_center = (self._rocket_pos(frame)
                                + np.array([0.0, 0.0, 0.35])
                                + self._pan_offset)
            self._apply_camera()
        return (self.body_poly, self.nozzle_poly, self.hmark, self.vmark,
                self.rmark)

    def _telemetry_text(self, frame):
        st = self.history[frame]
        d = flight_telemetry(st, self.cfg)
        ox, oy, oz = d["thrust_application_point_m"]
        ft = d["force_thrust_N"]
        fdirs = d["force_thrust_dir_world"]
        fdir_s = np.array2string(fdirs, precision=2, suppress_small=True)
        return "\n".join([
            f"t        = {d['time_s']:7.2f} s",
            f"pos      = X {d['position_x_m']:6.2f}  Y {d['position_y_m']:6.2f}  Z {d['position_z_m']:6.2f} m",
            f"height   = {d['height_m']:7.2f} m",
            f"vel      = X {d['velocity_x_m_s']:6.2f}  Y {d['velocity_y_m_s']:6.2f}  Z {d['velocity_z_m_s']:6.2f} m/s",
            f"speed    = {d['speed_m_s']:7.2f} m/s",
            f"speed_h  = {d['speed_horizontal_m_s']:7.2f} m/s  (Seitlich, X/Y)",
            f"acc      = X {d['acceleration_x_m_s2']:6.2f}  Y {d['acceleration_y_m_s2']:6.2f}  Z {d['acceleration_z_m_s2']:6.2f} m/s2",
            f"|acc|    = {d['acceleration_mag_m_s2']:7.2f} m/s2",
            "",
            f"q_B->I   = {np.array2string(d['quaternion_B_to_I'], precision=3, suppress_small=True)}",
            f"Roll/Pit/Yaw = {np.degrees(quat_to_euler(d['quaternion_B_to_I']))} deg",
            f"omega_b  = {np.array2string(d['omega_b_rad_s'], precision=3)} rad/s",
            f"|omega_b|= {d['omega_b_mag_rad_s']:7.3f} rad/s",
            f"alpha_b  = {np.array2string(d['alpha_b_rad_s2'], precision=3)} rad/s2",
            f"|alpha_b|= {d['alpha_b_mag_rad_s2']:7.3f} rad/s2",
            "",
            f"Trocken   = {d['dry_mass_kg']:7.1f} kg",
            f"Treibstoff= {d['fuel_kg']:7.1f} kg"
            + ("" if d["fuel_modeled"] else "  (nicht modelliert)"),
            f"Gesamt    = {d['total_mass_kg']:7.1f} kg",
            "",
            f"Schub     = {d['thrust_N']:7.1f} N   "
            f"Steuerung = {'MANUELL' if d['servo_manual'] else 'AUTO (Regler)'}",
            f" Servo X  = Soll {np.degrees(d['gimbal_x_cmd_rad']):+5.1f}  "
            f"Ist {np.degrees(d['gimbal_x_rad']):+5.1f} deg  "
            f"rate {np.degrees(d['servo_rate_x_rad_s']):+6.1f} deg/s"
            + ("  [SAT]" if d["servo_x_saturated"] else ""),
            f" Servo Y  = Soll {np.degrees(d['gimbal_y_cmd_rad']):+5.1f}  "
            f"Ist {np.degrees(d['gimbal_y_rad']):+5.1f} deg  "
            f"rate {np.degrees(d['servo_rate_y_rad_s']):+6.1f} deg/s"
            + ("  [SAT]" if d["servo_y_saturated"] else ""),
            f" F_Schub  = {ft[0]:7.1f} {ft[1]:7.1f} {ft[2]:7.1f} N",
            f"  |F|     = {d['force_thrust_mag_N']:7.1f} N",
            f"  Richtung= {fdir_s}",
            f"  Angriff = ({ox:.2f}, {oy:.2f}, {oz:.2f}) m (Doensenau)",
            f" F_Schw.  = {d['force_gravity_N'][0]:7.1f} {d['force_gravity_N'][1]:7.1f} "
            f"{d['force_gravity_N'][2]:7.1f} N  |F| = {d['force_gravity_mag_N']:.1f} N",
            f" F_gesamt = {d['force_total_N'][0]:7.1f} {d['force_total_N'][1]:7.1f} "
            f"{d['force_total_N'][2]:7.1f} N",
            f"  |F_ges.  = {d['force_total_mag_N']:7.1f} N",
            "",
            f"Moment    = {d['moment_total_body_Nm'][0]:6.2f} {d['moment_total_body_Nm'][1]:6.2f} "
            f"{d['moment_total_body_Nm'][2]:6.2f} Nm (Koerper)",
            f" |M|      = {d['moment_total_mag_Nm']:6.2f} Nm",
            "",
            f"Modell: {d['thrust_model_note']}",
            "Schubmodell: vereinfacht/virtuell",
            "Schwerkraft: ja (g0 = 9.80665 m/s2)",
            "Luftwiderstand / Aerodynamik: NICHT MODELLIERT",
            "Reale Triebwerkssteuerung: NICHT MODELLIERT",
            "",
            f"Phase: {self._live_status()}",
        ])

    def _live_status(self):
        if not self.history:
            return "NO_DATA"
        if self.frame >= len(self.history) - 1:
            return "ENDE"
        if self.paused:
            return "PAUSIERT"
        return "LAEUFT"

    # ------------------------------------------------------------- event hooks
    def _on_scroll(self, event):
        if event.inaxes is not self.ax3d:
            return
        step = getattr(event, "step", 0) or 0
        self.zoom(0.85 if step > 0 else 1.18)

    def _on_press(self, event):
        if event.inaxes is not self.ax3d or event.button is None:
            return
        self._drag = (int(event.button), float(event.x), float(event.y))

    def _on_motion(self, event):
        if self._drag is None or event.inaxes is not self.ax3d:
            return
        btn, x0, y0 = self._drag
        dx, dy = event.x - x0, event.y - y0
        self._drag = (btn, float(event.x), float(event.y))
        if btn == 1:
            self.rotate(-dx * 0.4, dy * 0.4)
        elif btn in (2, 3):
            self.pan(dx, dy)

    def _on_release(self, _event):
        self._drag = None

    def _on_key(self, event):
        key = event.key
        if key in ("+", "=", "pageup"):
            self.zoom(0.8)
        elif key in ("-", "_", "pagedown"):
            self.zoom(1.25)
        elif key == "o":
            self.set_view_mode(VIEW_OVERVIEW)
        elif key == "c":
            self.set_view_mode(VIEW_CLOSEUP)
        elif key == "r":
            self.reset_view()

    # ----------------------------------------------------------------- controls
    def set_dt(self, text):
        try:
            dt = float(text)
            if dt <= 0:
                raise ValueError
        except Exception:
            print(f"Ungueltiger dt: {text}")
            return
        self.cfg.dt = dt
        self.restart()

    def restart(self):
        """Recompute a fresh run from the current config/dt (from t0)."""
        self.history = run_flight(self.cfg)
        self.frame = 0
        self._cache_arrays()
        self._reset_axes()
        self.update(0)
        self._ensure_animation()
        self.paused = True
        self.b_pause.label.set_text("Weiter")
        self.play()

    def _on_mode(self, label):
        """Change the initial ATTITUDE only; both run the SAME shared simulation.

        'Geneigt' starts tilted so the coupling (thrust direction depends on q)
        and the model controller's self-righting are visible. Rotation is never
        a separate fake overlay.
        """
        if label == MODE_TILTED:
            a = np.deg2rad(15.0)
            self.cfg.start_q = np.array([np.cos(a / 2.0), 0.0,
                                         np.sin(a / 2.0), 0.0])
        else:
            self.cfg.start_q = np.array([1.0, 0.0, 0.0, 0.0])
        self.restart()

    def _on_view_mode(self, label):
        self.set_view_mode(label)

    # -------------------------------------------------------- manual servos
    def _on_ctrl_mode(self, label):
        """Switch between automatic control (the virtual controller) and manual
        servo control (the two sliders). The simulation is RECOMPUTED and the
        change is stated on screen, so the recompute is never silent."""
        self.cfg.manual_servo = (label == CTRL_MANUAL)
        if self.cfg.manual_servo:
            self.cfg.manual_gimbal_x_rad = float(np.deg2rad(self.slider_sx.val))
            self.cfg.manual_gimbal_y_rad = float(np.deg2rad(self.slider_sy.val))
            self._status_note = "MANUELL aktiv - Simulation neu berechnet (t=0)"
        else:
            self._status_note = "AUTO aktiv - Regler uebernimmt (neu berechnet)"
        self.restart()

    def _on_servo_x(self, val):
        if not self.cfg.manual_servo:
            return
        self.cfg.manual_gimbal_x_rad = float(np.deg2rad(float(val)))
        self._status_note = "MANUELL: Servo X geaendert - neu berechnet (t=0)"
        self.restart()

    def _on_servo_y(self, val):
        if not self.cfg.manual_servo:
            return
        self.cfg.manual_gimbal_y_rad = float(np.deg2rad(float(val)))
        self._status_note = "MANUELL: Servo Y geaendert - neu berechnet (t=0)"
        self.restart()

    def _control_status_text(self, frame=None):
        frame = self.frame if frame is None else frame
        mode = CTRL_MANUAL if self.cfg.manual_servo else CTRL_AUTO
        lines = [f"Modus : {mode}",
                 f"Grenzen: Servo +-{np.degrees(self.cfg.gimbal_max_rad):.1f} deg, "
                 f"Rate {np.degrees(self.cfg.servo_rate_max_rad_s):.1f} deg/s"]
        if self.history:
            st = self.history[min(max(frame, 0), len(self.history) - 1)]
            lines.append(
                f"Soll  : X {np.degrees(st.gimbal_x_cmd):+5.1f}  "
                f"Y {np.degrees(st.gimbal_y_cmd):+5.1f} deg (nur Sollwert)")
            lines.append(
                f"Ist   : X {np.degrees(st.gimbal_x):+5.1f}  "
                f"Y {np.degrees(st.gimbal_y):+5.1f} deg (wirkt auf Physik)")
        if self._status_note:
            lines.append(self._status_note)
        return "\n".join(lines)

    def _ensure_animation(self):
        """Create the ONE persistent animation (built once, never recreated).

        Matplotlib requires the `FuncAnimation` to stay referenced: dropping it
        garbage-collects the animation. On the macOS backend the animation's
        timer is a C-level CFRunLoop timer, so recreating/dropping an animation
        on every restart frees a still-registered timer -> use-after-free
        ("'TimerMac' object has no attribute 'callbacks'"). Creating it once and
        only resetting the frame counter on restart avoids this entirely.
        """
        if self.ani is not None or not self.history:
            return
        self.ani = FuncAnimation(self.fig, self._tick,
                                 frames=itertools.count(), interval=20,
                                 blit=False, repeat=False,
                                 cache_frame_data=False)

    def _tick(self, _frame_number):
        """Advance one frame per timer tick (loops back to the start at the end)."""
        n = len(self.history)
        if n == 0:
            return ()
        if not self.paused:
            self.frame = 0 if self.frame >= n - 1 else self.frame + 1
        return self.update(self.frame)

    def play(self):
        if not self.history:
            self.restart()
            return
        self._ensure_animation()
        self.paused = False
        self.b_pause.label.set_text("Pause")
        try:
            self.ani.event_source.start()
        except Exception:
            pass

    def toggle_pause(self):
        if not self.history:
            return
        if self.paused:
            self.play()
        else:
            self.paused = True
            self.b_pause.label.set_text("Weiter")
            try:
                self.ani.event_source.stop()
            except Exception:
                pass

    def show(self):
        self.play()
        plt.show()
