#!/usr/bin/env python3
"""
Interactive landing studio (learning simulation, simplified point-mass model).

Presentation layer only. It calls
  - simulation/scenario.py  (run_scenario)     -> physics/control orchestration
  - evaluation/landing.py   (evaluate_landing) -> landing assessment
and never contains its own physics/control logic.

Features
  - Scenario settings via text boxes: start height, horizontal start offset,
    horizontal target, initial vertical/horizontal velocity, time step.
  - Controls: Start, Pause/Resume, Reset, Recompute (new run every time).
  - Simultaneous results: 3D path + target, current height / vertical speed /
    horizontal speed / distance / elapsed time / control status, the final
    landing result of the whole run, and time-series plots (height, speed,
    distance vs time).
  - Current-frame values and whole-run results are clearly labelled.
  - A 3D rocket (nose + cylindrical body) with a separate engine nozzle bell.
    In "Rotation" mode the nozzle shows a purely VISUAL gimbal demo (no engine
    or thrust model exists).

NOT for real hardware. No real-rocket control.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, RadioButtons

from simulation.scenario import Scenario, run_scenario, run_rotation_demo
from evaluation import landing as landing_eval
from visualization.rocket_body import (
    rocket_geometry, nozzle_geometry, transform_points, rotate_about_axis,
    body_axis_world, nozzle_axis_world)

ROCKET_LENGTH_M = 1.0
NOZZLE_LENGTH_M = 0.26
GIMBAL_AMP_RAD = np.deg2rad(20.0)
GIMBAL_PERIOD_S = 2.0
MODE_LANDING = "Landung"
MODE_ROTATION = "Rotation"


class LandingStudio:
    def __init__(self, scenario: Scenario = None, max_display_frames: int = 1200,
                 history: list = None):
        self.scenario = scenario or Scenario()
        self.max_time_s = self.scenario.max_time_s  # not exposed in the UI
        self.max_display_frames = max_display_frames

        self.history = []
        self.result = None
        self.frame = 0
        self.paused = True
        self.ani = None
        self.mode = MODE_LANDING

        # cached per-run arrays
        self._t = self._z = self._vz = self._vxy = self._dist = None
        self._x = self._y = None

        # strings kept for tests / display
        self._live_text = ""
        self._result_text = ""
        self._message = ""

        self._boxes = {}
        self._build_figure()
        self._build_controls()
        self._build_3d_scene()
        self._build_series()
        if history is not None:
            # Adopt an already computed history; do NOT run/overwrite it.
            self._adopt_history(history, self.scenario.target())
        else:
            self.run_simulation()  # initial run with default scenario

    # ------------------------------------------------------------------ UI
    def _build_figure(self):
        self.fig = plt.figure(figsize=(15.0, 9.0))
        self.fig.suptitle(
            "Autoland Lern-Studio - vereinfachtes Punktmassen-Modell "
            "(keine reale Rakete)", fontsize=12)
        self.ax3d = self.fig.add_axes([0.03, 0.30, 0.44, 0.63], projection="3d")
        self.ax_h = self.fig.add_axes([0.55, 0.73, 0.42, 0.19])
        self.ax_v = self.fig.add_axes([0.55, 0.51, 0.42, 0.19])
        self.ax_d = self.fig.add_axes([0.55, 0.29, 0.42, 0.19])
        self.msg = self.fig.text(0.5, 0.005, "", ha="center", fontsize=9, color="0.25")

    def _textbox(self, key, label, value, x, y, w=0.072, h=0.036):
        ax = self.fig.add_axes([x, y, w, h])
        tb = TextBox(ax, "", initial=str(value))
        tb.on_submit(self._on_submit)
        self._boxes[key] = tb
        self.fig.text(x, y + h + 0.004, label, fontsize=8)
        return tb

    def _button(self, label, x, w=0.095):
        ax = self.fig.add_axes([x, 0.03, w, 0.045])
        return Button(ax, label)

    def _build_controls(self):
        sc = self.scenario
        y1, y2 = 0.19, 0.11
        self._textbox("h0", "Start-Hoehe [m]", sc.start_height_m, 0.05, y1)
        self._textbox("ox", "Offset X [m]", sc.start_offset_xy[0], 0.17, y1)
        self._textbox("oy", "Offset Y [m]", sc.start_offset_xy[1], 0.29, y1)
        self._textbox("tx", "Ziel X [m]", sc.target_xy[0], 0.41, y1)
        self._textbox("ty", "Ziel Y [m]", sc.target_xy[1], 0.05, y2)
        self._textbox("vz", "v0 vertikal [m/s]", sc.v0_vertical_m_s, 0.17, y2)
        self._textbox("vx", "v0 horiz (+X) [m/s]", sc.v0_horizontal_m_s, 0.29, y2)
        self._textbox("dt", "dt [s]", sc.dt, 0.41, y2)

        self.b_start = self._button("Start", 0.55)
        self.b_pause = self._button("Pause", 0.665)
        self.b_reset = self._button("Reset", 0.78)
        self.b_recompute = self._button("Neu berechnen", 0.55, w=0.21)
        self.b_recompute.ax.set_position([0.55, 0.03, 0.21, 0.045])

        self.b_start.on_clicked(self._on_start)
        self.b_pause.on_clicked(self._on_pause)
        self.b_reset.on_clicked(self._on_reset)
        self.b_recompute.on_clicked(self._on_recompute)

        # Mode selector: "Landung" (default, unchanged) vs. "Rotation" demo.
        ax_mode = self.fig.add_axes([0.05, 0.225, 0.14, 0.06])
        self.mode_radio = RadioButtons(ax_mode, (MODE_LANDING, MODE_ROTATION),
                                       active=0)
        self.mode_radio.on_clicked(self._on_mode_change)
        self.fig.text(0.05, 0.29, "Modus", fontsize=8)

    def _on_submit(self, _text):
        self._message = "Einstellungen geaendert - 'Neu berechnen' druecken."
        self.msg.set_text(self._message)

    # ----------------------------------------------------------- 3D scene
    def _build_3d_scene(self):
        self.ax3d.set_xlabel("X [m]")
        self.ax3d.set_ylabel("Y [m]")
        self.ax3d.set_zlabel("Z / Hoehe [m]")
        self._ground = None
        self._axes_arrows = []
        self.trail, = self.ax3d.plot([], [], [], "k-", alpha=0.5, lw=1.2)

        # 3D rocket: separate body and engine-nozzle geometries.
        self.body_verts, self.body_faces = rocket_geometry(length=ROCKET_LENGTH_M)
        self.nozzle_verts, self.nozzle_faces = nozzle_geometry(length=NOZZLE_LENGTH_M)
        self._body_world = transform_points(self.body_verts, [1, 0, 0, 0], np.zeros(3))
        self._nozzle_world = transform_points(self.nozzle_verts, [1, 0, 0, 0], np.zeros(3))
        self.gimbal_angle = 0.0
        self.body_poly = self._make_poly(self._body_world, self.body_faces,
                                         color="#c8ccd2", edge="#5a5f66")
        self.nozzle_poly = self._make_poly(self._nozzle_world, self.nozzle_faces,
                                           color="#6b7078", edge="#2f3338")
        self.tpt, = self.ax3d.plot([], [], [], "bx", ms=12, mew=2, label="Ziel")
        self.ax3d.legend(loc="upper left")
        self.joint_text = self.ax3d.text2D(0.02, 0.86, "",
                                           transform=self.ax3d.transAxes,
                                           va="top", fontsize=8, color="darkgreen")
        self.live = self.ax3d.text2D(0.02, 0.98, "", transform=self.ax3d.transAxes,
                                     va="top", fontsize=9, family="monospace")
        self.final = self.ax3d.text2D(0.98, 0.98, "", transform=self.ax3d.transAxes,
                                      va="top", ha="right", fontsize=9, color="darkred")

    def _make_poly(self, world_verts, faces, color, edge):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        polys = [world_verts[list(f)] for f in faces]
        coll = Poly3DCollection(polys, facecolor=color, edgecolor=edge,
                                linewidths=0.4, alpha=1.0)
        self.ax3d.add_collection3d(coll)
        return coll

    def _update_rocket(self, q, pos, gimbal_angle):
        """Update body + nozzle geometry for the given frame (world coords)."""
        self.gimbal_angle = float(gimbal_angle)
        self._body_world = transform_points(self.body_verts, q, pos)
        nozzle_body = rotate_about_axis(self.nozzle_verts, (0.0, 1.0, 0.0),
                                        gimbal_angle, origin=(0.0, 0.0, 0.0))
        self._nozzle_world = transform_points(nozzle_body, q, pos)
        self.body_poly.set_verts([self._body_world[list(f)] for f in self.body_faces])
        self.nozzle_poly.set_verts([self._nozzle_world[list(f)] for f in self.nozzle_faces])

    def rocket_tip_world(self):
        """World position of the nose tip (for tests / programmatic use)."""
        return self._body_world[-2].copy()

    def body_world(self):
        """Current world vertices of the rocket body (for tests)."""
        return self._body_world.copy()

    def nozzle_world(self):
        """Current world vertices of the engine nozzle (for tests)."""
        return self._nozzle_world.copy()

    def body_axis_world(self):
        """Current world direction of the rocket long axis (base -> tip)."""
        return body_axis_world(self._q[self.frame])

    def nozzle_axis_world(self):
        """Current world direction of the nozzle axis (with demo gimbal)."""
        return nozzle_axis_world(self._q[self.frame], self.gimbal_angle)

    def _reconfigure_3d(self, target):
        xmin = float(min(self._x.min(), target[0])) - 2.0
        xmax = float(max(self._x.max(), target[0])) + 2.0
        ymin = float(min(self._y.min(), target[1])) - 2.0
        ymax = float(max(self._y.max(), target[1])) + 2.0
        zmax = float(max(1.0, self._z.max()))
        self.ax3d.set_xlim(xmin, xmax)
        self.ax3d.set_ylim(ymin, ymax)
        self.ax3d.set_zlim(0.0, zmax)

        if self._ground is not None:
            self._ground.remove()
        for a in self._axes_arrows:
            a.remove()
        self._axes_arrows = []
        xx, yy = np.meshgrid([xmin, xmax], [ymin, ymax])
        self._ground = self.ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                                              alpha=0.2, color="green")
        self._axes_arrows.append(self.ax3d.quiver(xmin, ymin, 0, (xmax - xmin) * 0.2, 0, 0, color="r"))
        self._axes_arrows.append(self.ax3d.quiver(xmin, ymin, 0, 0, (ymax - ymin) * 0.2, 0, color="b"))
        self._axes_arrows.append(self.ax3d.quiver(xmin, ymin, 0, 0, 0, zmax * 0.2, color="g"))
        self.tpt.set_data([target[0]], [target[1]])
        self.tpt.set_3d_properties([0.0])
        # better camera view
        self.ax3d.view_init(elev=25.0, azim=135.0)

    # -------------------------------------------------------- time series
    def _build_series(self):
        for ax, title, ylab in (
                (self.ax_h, "Hoehe ueber Zeit", "h [m]"),
                (self.ax_v, "Geschwindigkeit ueber Zeit", "v [m/s]"),
                (self.ax_d, "Zielabstand ueber Zeit", "dist [m]")):
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("t [s]", fontsize=8)
            ax.set_ylabel(ylab, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        self.hline, = self.ax_h.plot([], [], "g-", alpha=0.6, lw=1.0)
        self.hmark, = self.ax_h.plot([], [], "go", ms=6)
        self.vline, = self.ax_v.plot([], [], "b-", alpha=0.6, lw=1.0)
        self.vmark, = self.ax_v.plot([], [], "bo", ms=6)
        self.vzline, = self.ax_v.plot([], [], "m--", alpha=0.5, lw=0.9, label="v_z")
        self.dline, = self.ax_d.plot([], [], "r-", alpha=0.6, lw=1.0)
        self.dmark, = self.ax_d.plot([], [], "ro", ms=6)
        self.ax_v.legend(loc="upper right", fontsize=7)
        self._artists = (self.trail, self.body_poly, self.nozzle_poly,
                         self.live, self.final, self.joint_text,
                         self.hmark, self.vmark, self.dmark)

    # ------------------------------------------------------------- running
    def _read_scenario(self) -> Scenario:
        f = lambda k: float(self._boxes[k].text)
        return Scenario(
            start_height_m=f("h0"),
            start_offset_xy=(f("ox"), f("oy")),
            target_xy=(f("tx"), f("ty")),
            v0_vertical_m_s=f("vz"),
            v0_horizontal_m_s=f("vx"),
            dt=f("dt"),
            max_time_s=self.max_time_s,
        )

    def run_simulation(self):
        """Recompute a fresh run from the current settings (never a replay)."""
        try:
            sc = self._read_scenario()
            hist = run_scenario(sc)
        except ValueError as exc:
            self._message = f"Ungueltige Einstellung: {exc}"
            self.msg.set_text(self._message)
            return None

        self.scenario = sc
        self.history = hist
        self.result = landing_eval.evaluate_landing(hist, sc.target())
        self.frame = 0
        self._cache_arrays()
        self._reconfigure_3d(sc.target())
        self._reset_series_axes()
        self._message = ("Neuer Versuch berechnet: "
                         f"{len(hist)} Schritte, Ergebnis {self.result['status']}.")
        self.msg.set_text(self._message)
        self._make_animation()
        self.update(0)
        return self.result

    def _adopt_history(self, history, target):
        """Display an already computed history (no scenario run, no overwrite)."""
        self.history = list(history)
        self.result = landing_eval.evaluate_landing(self.history, np.asarray(target, float))
        self.frame = 0
        self._cache_arrays()
        self._reconfigure_3d(np.asarray(target, float))
        self._reset_series_axes()
        self._message = (f"Vorhandene History geladen: {len(self.history)} Schritte, "
                         f"Ergebnis {self.result['status']}.")
        self.msg.set_text(self._message)
        self._make_animation()
        self.update(0)
        return self.result

    # ------------------------------------------------------ rotation demo mode
    def _on_mode_change(self, label):
        if label == MODE_ROTATION:
            self.mode = MODE_ROTATION
            self._load_rotation_mode()
        else:
            self.mode = MODE_LANDING
            self.run_simulation()

    def _load_rotation_mode(self):
        """Load the virtual rotation demo (separate from the landing run).

        This is a VISUAL DEMO ONLY: we give the nozzle a simple, periodic
        gimbal (sine) to make the relative motion clearly visible. There is no
        claim that this models a real engine/thrust/nozzle controller.
        """
        hist = run_rotation_demo()
        self._adopt_history(hist, self.scenario.target())
        self._frame_rotation_camera()
        self._message = (f"Rotations-Demo (visuelle Gimbal-Demo): {len(hist)} Schritte.")
        self.msg.set_text(self._message)
        return self.result

    def _frame_rotation_camera(self):
        """Zoom the 3D view onto the (nearly stationary) rotating rocket."""
        cx = float(np.mean(self._x))
        cy = float(np.mean(self._y))
        cz = float(np.mean(self._z))
        span = ROCKET_LENGTH_M * 1.6
        self.ax3d.set_xlim(cx - span, cx + span)
        self.ax3d.set_ylim(cy - span, cy + span)
        self.ax3d.set_zlim(cz - span, cz + span)
        self.ax3d.view_init(elev=18.0, azim=-60.0)
        try:
            self.ax3d.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    def _cache_arrays(self):
        self._x = np.array([s.pos[0] for s in self.history])
        self._y = np.array([s.pos[1] for s in self.history])
        self._z = np.array([s.height_m for s in self.history])
        self._t = np.array([s.t for s in self.history])
        self._q = np.array([np.asarray(s.q, dtype=float) for s in self.history])
        self._vz = np.array([s.vel[2] for s in self.history])
        self._vxy = np.array([float(np.hypot(s.vel[0], s.vel[1])) for s in self.history])
        tx, ty = self.scenario.target_xy
        self._dist = np.hypot(self._x - tx, self._y - ty)

    def _reset_series_axes(self):
        tmax = max(self._t.max(), 1e-6)
        for ax in (self.ax_h, self.ax_v, self.ax_d):
            ax.set_xlim(0.0, tmax)
        self.ax_h.set_ylim(0.0, max(1.0, self._z.max() * 1.05))
        self.ax_v.set_ylim(0.0, max(1.0, self._vxy.max() * 1.05))
        self.ax_d.set_ylim(0.0, max(1.0, self._dist.max() * 1.05))
        self.hline.set_data(self._t, self._z)
        self.vline.set_data(self._t, self._vxy)
        self.vzline.set_data(self._t, np.abs(self._vz))
        self.dline.set_data(self._t, self._dist)
        self._vline_artist = None  # not used; markers + full lines suffice

    def _make_animation(self):
        if self.ani is not None:
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
        n = len(self.history)
        if n == 0:
            self.ani = None
            return
        stride = max(1, n // self.max_display_frames)
        frames = list(range(0, n, stride))
        if frames[-1] != n - 1:
            frames.append(n - 1)
        self.ani = FuncAnimation(self.fig, self.update, frames=frames,
                                 interval=20, blit=False, repeat=False,
                                 cache_frame_data=False)

    # ------------------------------------------------------------ display
    def _live_status(self):
        if not self.history or self.result is None:
            return "NO_DATA"
        if self.frame >= len(self.history) - 1:
            return self.result["status"]
        return "IN_FLIGHT"

    def _refresh_text(self):
        if not self.history:
            self.live.set_text("kein Versuch")
            self.final.set_text("")
            if getattr(self, "joint_text", None) is not None:
                self.joint_text.set_text("")
            return
        f = self.frame
        last = len(self.history) - 1
        self._live_text = (
            f"AKTUELL (Frame {f + 1}/{len(self.history)})\n"
            f"t    = {self._t[f]:6.2f} s\n"
            f"h    = {self._z[f]:6.2f} m\n"
            f"v_z  = {self._vz[f]:6.2f} m/s\n"
            f"v_xy = {self._vxy[f]:6.2f} m/s\n"
            f"dist = {self._dist[f]:6.2f} m\n"
            f"Regelung: AKTIV\n"
            f"Phase: {self._live_status()}"
        )
        self.live.set_text(self._live_text)

        r = self.result
        tag = "GESAMTERGEBNIS (berechnet)" if r else "kein Ergebnis"
        if r:
            self._result_text = (
                f"{tag}\n"
                f"Status   = {r['status']}\n"
                f"Kontakt  = {'ja' if r.get('contact') else 'nein'}\n"
                f"Treffer  = {'ja' if r.get('on_target') else 'nein'}\n"
                f"Abstand  = {r.get('dist_xy', float('nan')):.2f} m\n"
                f"Geschw.  = {r.get('v_mag_contact', float('nan')):.2f} m/s"
            )
        else:
            self._result_text = tag
        self.final.set_text(self._result_text)
        if getattr(self, "joint_text", None) is not None:
            ang = np.rad2deg(self.gimbal_angle)
            self.joint_text.set_text(
                f"Düse (visuelle Demo): Gimbal {ang:+.1f}°\n"
                "Triebwerk: KEIN Schubmodell")

    def update(self, frame):
        if not self.history:
            return ()
        frame = int(min(max(frame, 0), len(self.history) - 1))
        self.frame = frame
        self.trail.set_data(self._x[:frame + 1], self._y[:frame + 1])
        self.trail.set_3d_properties(self._z[:frame + 1])
        q = self._q[frame]
        pos = np.array([self._x[frame], self._y[frame], self._z[frame]])
        gimbal = GIMBAL_AMP_RAD * np.sin(2.0 * np.pi * self._t[frame] / GIMBAL_PERIOD_S) if self.mode == MODE_ROTATION else 0.0
        self._update_rocket(q, pos, gimbal)
        t = self._t[frame]
        self.hmark.set_data([t], [self._z[frame]])
        self.vmark.set_data([t], [self._vxy[frame]])
        self.dmark.set_data([t], [self._dist[frame]])
        self._refresh_text()
        return self._artists

    def readout(self) -> dict:
        """Current displayed values (for tests / programmatic use)."""
        if not self.history:
            return {}
        f = self.frame
        return {
            "frame": f,
            "n_frames": len(self.history),
            "t": float(self._t[f]),
            "height": float(self._z[f]),
            "v_vertical": float(self._vz[f]),
            "v_horizontal": float(self._vxy[f]),
            "distance": float(self._dist[f]),
            "live_status": self._live_status(),
            "final_status": self.result["status"] if self.result else None,
            "live_text": self._live_text,
            "result_text": self._result_text,
        }

    # ----------------------------------------------------------- controls
    def set_setting(self, key, value):
        if key not in self._boxes:
            raise KeyError(key)
        self._boxes[key].set_val(str(value))

    def _on_start(self, _event=None):
        self.play()

    def _on_recompute(self, _event=None):
        if self.mode == MODE_ROTATION:
            self._load_rotation_mode()
        else:
            self.run_simulation()
        self.play()

    def _on_pause(self, _event=None):
        self.toggle_pause()

    def _on_reset(self, _event=None):
        self.reset_run()

    def play(self):
        if not self.history:
            if self.run_simulation() is None:
                return
        if self.frame >= len(self.history) - 1:
            self.update(0)
            self._make_animation()
        self.paused = False
        self.b_pause.label.set_text("Pause")
        if self.ani is not None:
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
            if self.ani is not None:
                try:
                    self.ani.event_source.stop()
                except Exception:
                    pass

    def reset_run(self):
        if not self.history:
            return
        self.paused = True
        self.b_pause.label.set_text("Weiter")
        self._make_animation()
        if self.ani is not None:
            try:
                self.ani.event_source.stop()
            except Exception:
                pass
        self.update(0)

    def show(self):
        self.play()
        plt.show()
