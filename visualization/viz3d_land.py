#!/usr/bin/env python3
"""
Interactive 3D viz for learning: shows rocket, ground, axes, target landing point.
Uses real simulation history from Simulator. Can change target X,Y in UI (sliders).
Shows t, h, v_mag, distance to target.
No autoland/control implemented (clearly marked).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider


class SimpleLandViz3D:
    def __init__(self, history, target_pos=(0.0, 0.0, 0.0)):
        self.history = history
        self.target = np.array([target_pos[0], target_pos[1], 0.0])
        self.paused = False
        self.fig = plt.figure(figsize=(10, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._init_ui()
        self.setup_plot()

    def _init_ui(self):
        axplay = plt.axes([0.6, 0.02, 0.08, 0.04])
        self.bplay = Button(axplay, 'Pause' if not self.paused else 'Play')
        self.bplay.on_clicked(self.toggle)
        axrst = plt.axes([0.7, 0.02, 0.08, 0.04])
        self.brst = Button(axrst, 'Restart')
        self.brst.on_clicked(self.restart)
        axsx = plt.axes([0.15, 0.02, 0.15, 0.015])
        self.sx = Slider(axsx, 'Target X', -10.0, 10.0, valinit=self.target[0])
        self.sx.on_changed(self._on_target)
        axsy = plt.axes([0.15, 0.045, 0.15, 0.015])
        self.sy = Slider(axsy, 'Target Y', -10.0, 10.0, valinit=self.target[1])
        self.sy.on_changed(self._on_target)

    def _on_target(self, val):
        self.target[0] = self.sx.val
        self.target[1] = self.sy.val
        if hasattr(self, 'target_pt'):
            self.target_pt.set_data([self.target[0]], [self.target[1]])
            self.target_pt.set_3d_properties([self.target[2]])

    def setup_plot(self):
        if not self.history:
            return
        xs = [h.pos[0] for h in self.history]
        ys = [h.pos[1] for h in self.history]
        zs = [h.pos[2] for h in self.history]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        zmin, zmax = min(zs), max(zs)
        if xmin == xmax: xmin, xmax = -5, 5
        if ymin == ymax: ymin, ymax = -5, 5
        if zmin == zmax: zmin, zmax = 0, 2
        # expand slightly for target
        xmin = min(xmin, self.target[0]-2); xmax = max(xmax, self.target[0]+2)
        ymin = min(ymin, self.target[1]-2); ymax = max(ymax, self.target[1]+2)
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set_zlim(0, max(1.0, zmax))
        xx, yy = np.meshgrid([xmin, xmax], [ymin, ymax])
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.2, color='green')
        self.ax.quiver(xmin, ymin, 0, (xmax-xmin)*0.25, 0, 0, color='r')
        self.ax.quiver(xmin, ymin, 0, 0, (ymax-ymin)*0.25, 0, color='b')
        self.ax.quiver(xmin, ymin, 0, 0, 0, max(1,zmax)*0.25, color='g')
        self.ax.set_xlabel('X (m)'); self.ax.set_ylabel('Y (m)'); self.ax.set_zlabel('Z (m, Up)')
        self.ax.set_title('Landing sim (real data; control NOT implemented)')
        self.trail, = self.ax.plot(xs[:1], ys[:1], zs[:1], 'k-', alpha=0.4, lw=1)
        self.rocket, = self.ax.plot(xs[0:1], ys[0:1], zs[0:1], 'ro', ms=6)
        self.target_pt, = self.ax.plot([self.target[0]], [self.target[1]], [self.target[2]], 'bx', ms=10, label='Landing target')
        self.ax.legend(loc='upper left')
        self.time_text = self.ax.text2D(0.02, 0.98, '', transform=self.ax.transAxes, va='top')
        self.h_text = self.ax.text2D(0.02, 0.94, '', transform=self.ax.transAxes, va='top')
        self.v_text = self.ax.text2D(0.02, 0.90, '', transform=self.ax.transAxes, va='top')
        self.d_text = self.ax.text2D(0.02, 0.86, '', transform=self.ax.transAxes, va='top')
        self.ani = FuncAnimation(self.fig, self.update, frames=len(self.history), interval=20, blit=False, repeat=True)

    def toggle(self, event=None):
        if self.paused:
            self.ani.event_source.start(); self.bplay.label.set_text('Pause'); self.paused=False
        else:
            self.ani.event_source.stop(); self.bplay.label.set_text('Play'); self.paused=True

    def restart(self, event=None):
        self.update(0)
        if self.paused: self.toggle()

    def update(self, frame):
        if frame < len(self.history):
            h = self.history[frame]
            xs=[x.pos[0] for x in self.history[:frame+1]]; ys=[x.pos[1] for x in self.history[:frame+1]]; zs=[x.pos[2] for x in self.history[:frame+1]]
            self.trail.set_data(xs,ys); self.trail.set_3d_properties(zs)
            self.rocket.set_data([h.pos[0]],[h.pos[1]]); self.rocket.set_3d_properties([h.pos[2]])
            vmag=float(np.linalg.norm(h.vel))
            d=float(np.linalg.norm(h.pos[:2]-self.target[:2]))
            self.time_text.set_text(f't = {h.t:.2f} s')
            self.h_text.set_text(f'h = {h.height_m:.1f} m')
            self.v_text.set_text(f'v = {vmag:.2f} m/s')
            self.d_text.set_text(f'dist(target) = {d:.2f} m')
        return self.trail,self.rocket,self.time_text,self.h_text,self.v_text,self.d_text

    def show(self):
        plt.show()


def run_landing_viz():
    from state.core import State
    from mass_model.mass import MassModel
    from simulation.loop import Simulator, SimConfig
    s = State(pos=[0.0,0.0,80.0]); m = MassModel(dry_mass_kg=300.0,fuel_kg=0.0)
    sim = Simulator(SimConfig(tf=4.0,dt=0.02)); hist = sim.run(s.copy(),m)
    viz = SimpleLandViz3D(hist, target_pos=(2.0,1.0,0.0)); viz.show()
