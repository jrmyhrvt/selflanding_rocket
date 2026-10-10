#!/usr/bin/env python3
"""
Simple 3D visualization using matplotlib (available).
Shows: ground, axes, rocket (small marker), trajectory trail.
Uses actual simulation history (no made-up data).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button


class SimpleRocketViz3D:
    def __init__(self, history):
        self.history = history
        self.paused = False
        self.fig = plt.figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.setup_plot()

    def setup_plot(self):
        if not self.history:
            return
        xs = [h.pos[0] for h in self.history]
        ys = [h.pos[1] for h in self.history]
        zs = [h.pos[2] for h in self.history]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
        if xmin == xmax: xmin, xmax = -1, 1
        if ymin == ymax: ymin, ymax = -1, 1
        if zmin == zmax: zmin, zmax = 0, 2
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(0, max(1.0, zmax))
        xx, yy = np.meshgrid([xmin, xmax], [ymin, ymax])
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='green')
        self.ax.quiver(xmin, ymin, 0, (xmax-xmin)*0.3, 0, 0, color='r', label='X')
        self.ax.quiver(xmin, ymin, 0, 0, (ymax-ymin)*0.3, 0, color='b', label='Y')
        self.ax.quiver(xmin, ymin, 0, 0, 0, max(1,zmax)*0.3, color='g', label='Z')
        self.ax.legend(loc='upper left')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m, Up)')
        self.ax.set_title('Rocket Trajectory (uses sim data)')
        self.trail, = self.ax.plot(xs[:1], ys[:1], zs[:1], 'k-', alpha=0.5, lw=1)
        self.rocket, = self.ax.plot(xs[0:1], ys[0:1], zs[0:1], 'ro', ms=6, label='Rocket')
        self.time_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes, va='top')
        self.h_text = self.ax.text2D(0.02, 0.94, '', transform=self.ax.transAxes, va='top')
        self.v_text = self.ax.text2D(0.02, 0.90, '', transform=self.ax.transAxes, va='top')
        axplay = plt.axes([0.7, 0.02, 0.08, 0.04])
        self.bplay = Button(axplay, 'Pause' if not self.paused else 'Play')
        self.bplay.on_clicked(self.toggle)
        axrst = plt.axes([0.8, 0.02, 0.08, 0.04])
        self.brst = Button(axrst, 'Restart')
        self.brst.on_clicked(self.restart)
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.history), interval=20, blit=False, repeat=True)

    def toggle(self, event=None):
        if self.paused:
            self.ani.event_source.start()
            self.bplay.label.set_text('Pause')
            self.paused = False
        else:
            self.ani.event_source.stop()
            self.bplay.label.set_text('Play')
            self.paused = True

    def restart(self, event=None):
        self.update(0)
        if self.paused:
            self.toggle()

    def update(self, frame):
        if frame < len(self.history):
            h = self.history[frame]
            xs = [x.pos[0] for x in self.history[:frame+1]]
            ys = [x.pos[1] for x in self.history[:frame+1]]
            zs = [x.pos[2] for x in self.history[:frame+1]]
            self.trail.set_data(xs, ys)
            self.trail.set_3d_properties(zs)
            self.rocket.set_data([h.pos[0]], [h.pos[1]])
            self.rocket.set_3d_properties([h.pos[2]])
            self.time_text.set_text(f't = {h.t:.2f} s')
            self.h_text.set_text(f'h = {h.height_m:.1f} m')
            vmag = float(np.linalg.norm(h.vel))
            self.v_text.set_text(f'v = {vmag:.1f} m/s')
        return self.trail, self.rocket, self.time_text, self.h_text, self.v_text

    def show(self):
        plt.show()


def run_visual():
    from state.core import State
    from mass_model.mass import MassModel
    from simulation.loop import Simulator, SimConfig
    s = State(pos=[0.0, 0.0, 80.0])
    m = MassModel(dry_mass_kg=300.0, fuel_kg=0.0)
    sim = Simulator(SimConfig(tf=4.0, dt=0.02))
    hist = sim.run(s.copy(), m)
    viz = SimpleRocketViz3D(hist)
    viz.show()
