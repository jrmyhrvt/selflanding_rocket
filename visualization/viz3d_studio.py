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

NOT for real hardware. No real-rocket control.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox

from simulation.scenario import Scenario, run_scenario
from evaluation import landing as landing_eval


class LandingStudio:
    def __init__(self, scenario: Scenario = None, max_display_frames: int = 1200):
        self.scenario = scenario or Scenario()
        self.max_time_s = self.scenario.max_time_s  # not exposed in the UI
        self.max_display_frames = max_display_frames

        self.history = []
        self.result = None
        self.frame = 0
        self.paused = True
        self.ani = None

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
        self.rock, = self.ax3d.plot([], [], [], "ro", ms=6)
        self.tpt, = self.ax3d.plot([], [], [], "bx", ms=12, mew=2, label="Ziel")
        self.ax3d.legend(loc="upper left")
        self.live = self.ax3d.text2D(0.02, 0.98, "", transform=self.ax3d.transAxes,
                                     va="top", fontsize=9, family="monospace")
        self.final = self.ax3d.text2D(0.98, 0.98, "", transform=self.ax3d.transAxes,
                                      va="top", ha="right", fontsize=9, color="darkred")

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
        self._artists = (self.trail, self.rock, self.live, self.final,
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

    def _cache_arrays(self):
        self._x = np.array([s.pos[0] for s in self.history])
        self._y = np.array([s.pos[1] for s in self.history])
        self._z = np.array([s.height_m for s in self.history])
        self._t = np.array([s.t for s in self.history])
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
                f"Status: {r['status']}\n"
                f"v_ende = {r['v_mag']:.2f} m/s\n"
                f"dist   = {r['dist_xy']:.2f} m\n"
                f"Kontakt: {'ja' if r.get('contact') else 'nein'}"
            )
        else:
            self._result_text = tag
        self.final.set_text(self._result_text)

    def update(self, frame):
        if not self.history:
            return ()
        frame = int(min(max(frame, 0), len(self.history) - 1))
        self.frame = frame
        self.trail.set_data(self._x[:frame + 1], self._y[:frame + 1])
        self.trail.set_3d_properties(self._z[:frame + 1])
        self.rock.set_data([self._x[frame]], [self._y[frame]])
        self.rock.set_3d_properties([self._z[frame]])
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
