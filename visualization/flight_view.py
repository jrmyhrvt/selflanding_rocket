"""Unified 3D flight view: ONE run, rocket + movable nozzle, force vectors,
live telemetry and plots.

Presentation only: every displayed number comes from the same simulation state
used for the animation (`simulation/flight.py`), read through
`flight_telemetry()`. Nothing is animated independently of the physics, and
quantities the model does not compute are explicitly labelled as not modelled.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button, RadioButtons, TextBox

from config import units
from physics.rotation import quat_to_euler
from simulation.flight import FlightConfig, flight_telemetry, run_flight
from visualization.rocket_body import (rocket_geometry, nozzle_geometry,
                                       transform_points, rotate_about_axis)

MODE_UPRIGHT = "Aufrecht"
MODE_TILTED = "Geneigt (15 Grad)"


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
        self._build_figure()
        self._cache_arrays()
        self._reset_axes()
        self.update(0)

    # ------------------------------------------------------------------ build
    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(3, 4, wspace=0.35, hspace=0.45)

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
        self._axes_arrows = []

        self.ax_tele = _fig_text_axis(self.fig, 0.505, 0.40, 0.24, 0.48,
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

        # ---- control widgets
        ax_start = self.fig.add_axes([0.05, 0.02, 0.09, 0.04])
        ax_pause = self.fig.add_axes([0.15, 0.02, 0.09, 0.04])
        ax_restart = self.fig.add_axes([0.25, 0.02, 0.09, 0.04])
        ax_dt = self.fig.add_axes([0.40, 0.02, 0.07, 0.04])
        ax_mode = self.fig.add_axes([0.53, 0.02, 0.13, 0.06])
        self.b_start = Button(ax_start, "Start")
        self.b_pause = Button(ax_pause, "Pause")
        self.b_restart = Button(ax_restart, "Neustart")
        self.box_dt = TextBox(ax_dt, "dt=", initial=str(self.cfg.dt))
        self.radios = RadioButtons(ax_mode, (MODE_UPRIGHT, MODE_TILTED))
        self.radios.set_active(0)

        self.b_start.on_clicked(lambda e: self.play())
        self.b_pause.on_clicked(lambda e: self.toggle_pause())
        self.b_restart.on_clicked(lambda e: self.restart())
        self.box_dt.on_submit(lambda text: self.set_dt(text))
        self.radios.on_clicked(self._on_mode)

        self._build_series()

    def _build_series(self):
        self.hline, = self.ax_h.plot([], [], "g-", alpha=0.7, lw=1.1)
        self.vline_mag, = self.ax_v.plot([], [], "b-", alpha=0.8, lw=1.1, label="|v|")
        self.vline_z, = self.ax_v.plot([], [], "m-", alpha=0.7, lw=1.0, label="v_z")
        self.vline_h, = self.ax_v.plot([], [], "c-", alpha=0.9, lw=1.1, label="|v_h|")
        self.aline_mag, = self.ax_v.plot([], [], "r--", alpha=0.7, lw=1.0, label="|a|")
        self.ax_v.legend(loc="upper right", fontsize=7)
        self.oline, = self.ax_r.plot([], [], "b-", alpha=0.8, lw=1.1, label="|omega_b|")
        self.gline_x, = self.ax_r.plot([], [], "g-", alpha=0.8, lw=1.1, label="gimbal_x [deg/10]")
        self.gline_y, = self.ax_r.plot([], [], "c-", alpha=0.8, lw=1.1, label="gimbal_y [deg/10]")
        self.ax_r.legend(loc="upper right", fontsize=7, ncol=3)
        self.hmark, = self.ax_h.plot([], [], "go", ms=5)
        self.vmark, = self.ax_v.plot([], [], "bo", ms=5)
        self.rmark, = self.ax_r.plot([], [], "bo", ms=5)

    # ------------------------------------------------------------------ arrays
    def _cache_arrays(self):
        h = self.history
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
        self._cache_forces = [(np.asarray(s.force_thrust, float),
                               np.asarray(s.force_gravity, float),
                               np.asarray(s.force_total, float)) for s in h]

    def _reset_axes(self):
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
                   float(np.abs(self._gim_y).max() * 180.0 / np.pi / 10.0))
        self.ax_r.set_ylim(-omax, omax)
        self.hline.set_data(self._t, self._z)
        self.vline_mag.set_data(self._t, np.linalg.norm(self._vel, axis=1))
        self.vline_z.set_data(self._t, self._vel[:, 2])
        self.vline_h.set_data(self._t, np.linalg.norm(self._vel[:, :2], axis=1))
        self.aline_mag.set_data(self._t, np.linalg.norm(self._acc, axis=1))
        self.oline.set_data(self._t, np.linalg.norm(self._omega_b, axis=1))
        self.gline_x.set_data(self._t, np.degrees(self._gim_x) / 10.0)
        self.gline_y.set_data(self._t, np.degrees(self._gim_y) / 10.0)

        # frame the 3D view around the whole trajectory
        cx, cy, cz = float(self._x.mean()), float(self._y.mean()), float(self._z.mean())
        span = max(float(np.ptp(self._x)), float(np.ptp(self._y)),
                   float(np.ptp(self._z)), 1.0)
        self.ax3d.set_xlim(cx - 0.6 * span, cx + 0.6 * span)
        self.ax3d.set_ylim(cy - 0.6 * span, cy + 0.6 * span)
        self.ax3d.set_zlim(0.0, max(1.0, cz + 0.6 * span))
        self.ax3d.view_init(elev=22.0, azim=-60.0)

        # ground plane + static world axes at the origin (drawn once)
        for a in self._axes_arrows:
            try:
                a.remove()
            except Exception:
                pass
        self._axes_arrows = []
        xx, yy = np.meshgrid([self.ax3d.get_xlim()[0], self.ax3d.get_xlim()[1]],
                             [self.ax3d.get_ylim()[0], self.ax3d.get_ylim()[1]])
        try:
            self._ground.remove()
        except Exception:
            pass
        self._ground = self.ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                                              alpha=0.15, color="green")
        o = np.array([self.ax3d.get_xlim()[0], self.ax3d.get_ylim()[0], 0.0])
        for k, c in enumerate(("r", "b", "g")):
            v = np.zeros(3)
            v[k] = span * 0.25
            self._axes_arrows.append(
                self.ax3d.quiver(o[0], o[1], o[2], v[0], v[1], v[2],
                                 color=c, linewidths=1.4, normalize=False))
        self._axes_arrows.append(self.ax3d.text(o[0], o[1], o[2] + span * 0.25,
                                                "Welt x/y/z", fontsize=7, color="k"))

    # ------------------------------------------------------------- geometry/3d
    def body_world(self):
        return self._body_world

    def nozzle_world(self):
        return self._nozzle_world

    def _rocket_world(self, frame):
        q = self._q[frame]
        pos = np.array([self._x[frame], self._y[frame], self._z[frame]])
        bv, bf = rocket_geometry()
        nv, nf = nozzle_geometry()
        nv = rotate_about_axis(nv, [1.0, 0.0, 0.0], self._gim_x[frame])
        nv = rotate_about_axis(nv, [0.0, 1.0, 0.0], self._gim_y[frame])
        return (transform_points(bv, q, pos), bf,
                transform_points(nv, q, pos), nf, q, pos)

    def _draw_rocket(self, frame):
        bw, bf, nw, nf, q, pos = self._rocket_world(frame)
        self._body_world, self._nozzle_world = bw, nw
        self.body_poly.set_verts([bw[fi] for fi in bf])
        self.nozzle_poly.set_verts([nw[fi] for fi in nf])

        # rocket body axes (frame) at the centre of mass, redrawn each frame
        self._clear_vectors()
        self._add_axes(pos, q, L=0.6)
        # labeled vectors: velocity, acceleration, forces
        self._add_vector(pos, self._vel[frame], color="cyan", label="v")
        self._add_vector(pos, self._acc[frame], color="magenta", label="a")
        self._add_vector(pos, self._cache_forces[frame][0], color="orange",
                         label="F_Schub")
        self._add_vector(pos, self._cache_forces[frame][1], color="gray",
                         label="F_Schwerkraft")
        self._add_vector(pos, self._cache_forces[frame][2], color="red",
                         label="F_gesamt")

    def _clear_vectors(self):
        for a in getattr(self, "_vec_arrows", []):
            try:
                a.remove()
            except Exception:
                pass
        self._vec_arrows = []
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
        from physics.rotation import rotate_vector
        colors = ("r", "b", "g")
        for k in range(3):
            e = np.zeros(3)
            e[k] = L
            v = rotate_vector(q, e)
            self._vec_arrows.append(
                self.ax3d.quiver(origin[0], origin[1], origin[2],
                                 v[0], v[1], v[2], color=colors[k],
                                 linewidths=1.2, normalize=False))
        tip = np.asarray(origin, float) + rotate_vector(q, np.array([0.0, 0.0, L]))
        self._vec_labels.append(
            self.ax3d.text(tip[0], tip[1], tip[2], "Koerper-Z", fontsize=7,
                           color="g"))

    def _add_vector(self, origin, vec, color, label, scale=1.0):
        v = np.asarray(vec, dtype=float) * scale
        if not np.all(np.isfinite(v)) or np.allclose(v, 0.0):
            return
        arrow = self.ax3d.quiver(origin[0], origin[1], origin[2],
                                 v[0], v[1], v[2], color=color,
                                 linewidths=1.6, normalize=False)
        self._vec_arrows.append(arrow)
        tip = np.asarray(origin, dtype=float) + v
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
        return (self.body_poly, self.nozzle_poly, self.hmark, self.vmark, self.rmark)

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
            f"Schub     = {d['thrust_N']:7.1f} N   Gimbal x/y = "
            f"{np.degrees(d['gimbal_x_rad']):+5.1f}/{np.degrees(d['gimbal_y_rad']):+5.1f} deg",
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
