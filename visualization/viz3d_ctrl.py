#!/usr/bin/env python3
"""
Learning viz: shows the ACTUAL simulated path, target point, distance, height,
speed and the guidance/landing status.

- If `guidance_active=True` the path was produced with an active (virtual)
  controller coupled to the simulator; the status is a live landing
  evaluation.
- If `guidance_active=False` the path is gravity-only (free fall) and the
  status is 'NO_CTRL'. No landing is fabricated.

Live re-simulation:
- If `resimulate` is given (a callable `target_xyz -> history`), moving the
  target sliders re-runs the simulation and reloads the scene, so the shown
  trajectory really is the current simulation (not a stale replay).
- If `resimulate is None` the scene only replays the given history; the
  target marker moves but the trajectory is not recomputed.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from evaluation import landing as landing_eval


class CtrlViz3D:
    def __init__(self, history, target_pos=(0.0,0.0,0.0), guidance_active=False,
                 resimulate=None):
        self.history = history
        self.target = np.array([target_pos[0],target_pos[1],0.0])
        self.guidance_active = bool(guidance_active)
        self.resimulate = resimulate
        self.paused=False
        self.fig=plt.figure(figsize=(10,7))
        self.ax=self.fig.add_subplot(111,projection='3d')
        self._ui(); self.setup()

    def _ui(self):
        axp=plt.axes([0.6,0.02,0.08,0.04]); self.bp=Button(axp,'Pause'); self.bp.on_clicked(self.toggle)
        axr=plt.axes([0.7,0.02,0.08,0.04]); self.br=Button(axr,'Restart'); self.br.on_clicked(self.restart)
        axs=plt.axes([0.15,0.02,0.15,0.015]); self.sx=Slider(axs,'Target X',-10,10,valinit=self.target[0]); self.sx.on_changed(self._t)
        axy=plt.axes([0.15,0.045,0.15,0.015]); self.sy=Slider(axy,'Target Y',-10,10,valinit=self.target[1]); self.sy.on_changed(self._t)

    def _t(self,v):
        self.target[0]=self.sx.val; self.target[1]=self.sy.val
        if self.resimulate is not None:
            new_hist = self.resimulate(self.target)
            if new_hist:
                self.history = new_hist
                self.reload()
                return
        if hasattr(self,'tpt'): self.tpt.set_data([self.target[0]],[self.target[1]]); self.tpt.set_3d_properties([0])

    def reload(self):
        """Stop the current animation, rebuild the scene for self.history."""
        if hasattr(self,'ani'):
            self.ani.event_source.stop()
        self.ax.clear()
        self.paused = False
        self.setup()

    def setup(self):
        if hasattr(self,'ani'):
            self.ani.event_source.stop()
        if not self.history: return
        xs=[h.pos[0] for h in self.history]; ys=[h.pos[1] for h in self.history]; zs=[h.pos[2] for h in self.history]
        xmin=min(xs); xmax=max(xs); ymin=min(ys); ymax=max(ys); zmax=max(zs)
        if xmin==xmax:xmin,xmax=-5,5
        if ymin==ymax:ymin,ymax=-5,5
        xmin=min(xmin,self.target[0]-2); xmax=max(xmax,self.target[0]+2); ymin=min(ymin,self.target[1]-2); ymax=max(ymax,self.target[1]+2)
        self.ax.set_xlim(xmin,xmax); self.ax.set_ylim(ymin,ymax); self.ax.set_zlim(0,max(1,zmax))
        xx,yy=np.meshgrid([xmin,xmax],[ymin,ymax]); self.ax.plot_surface(xx,yy,np.zeros_like(xx),alpha=0.2,color='green')
        self.ax.quiver(xmin,ymin,0,(xmax-xmin)*0.25,0,0,color='r'); self.ax.quiver(xmin,ymin,0,0,(ymax-ymin)*0.25,0,color='b'); self.ax.quiver(xmin,ymin,0,0,0,max(1,zmax)*0.25,color='g')
        ctrl_txt = 'ON (coupled)' if self.guidance_active else 'OFF (free fall)'
        self.ax.set_title(f'Autoland (learning, simplified): real sim data; CTRL {ctrl_txt}')
        self.trail,=self.ax.plot(xs[:1],ys[:1],zs[:1],'k-',alpha=0.4); self.rock,=self.ax.plot(xs[0:1],ys[0:1],zs[0:1],'ro',ms=6)
        self.tpt,=self.ax.plot([self.target[0]],[self.target[1]],[0],'bx',ms=10,label='Target')
        self.ax.legend(loc='upper left')
        self.tt=self.ax.text2D(0.02,0.98,'',transform=self.ax.transAxes,va='top')
        self.ht=self.ax.text2D(0.02,0.94,'',transform=self.ax.transAxes,va='top')
        self.vt=self.ax.text2D(0.02,0.90,'',transform=self.ax.transAxes,va='top')
        self.dt=self.ax.text2D(0.02,0.86,'',transform=self.ax.transAxes,va='top')
        self.ct=self.ax.text2D(0.02,0.82,'CTRL: '+('ON' if self.guidance_active else 'OFF'),transform=self.ax.transAxes,va='top')
        self.st=self.ax.text2D(0.02,0.78,'',transform=self.ax.transAxes,va='top')
        self.ani=FuncAnimation(self.fig,self.update,frames=len(self.history),interval=20,blit=False,repeat=True)

    def toggle(self,e=None):
        if self.paused: self.ani.event_source.start(); self.bp.label.set_text('Pause'); self.paused=False
        else: self.ani.event_source.stop(); self.bp.label.set_text('Play'); self.paused=True

    def restart(self,e=None):
        self.update(0)
        if self.paused: self.toggle()

    def update(self,frame):
        if frame<len(self.history):
            h=self.history[frame]
            xs=[x.pos[0] for x in self.history[:frame+1]]; ys=[x.pos[1] for x in self.history[:frame+1]]; zs=[x.pos[2] for x in self.history[:frame+1]]
            self.trail.set_data(xs,ys); self.trail.set_3d_properties(zs)
            self.rock.set_data([h.pos[0]],[h.pos[1]]); self.rock.set_3d_properties([h.pos[2]])
            vmag=float(np.linalg.norm(h.vel)); d=float(np.linalg.norm(h.pos[:2]-self.target[:2]))
            if self.guidance_active:
                # Live landing evaluation on the path so far.
                res = landing_eval.evaluate_landing(self.history[:frame+1], self.target)
                status = res['status'] if res.get('contact') else 'IN_FLIGHT'
            else:
                status = 'NO_CTRL'
            self.tt.set_text(f't={h.t:.2f}s'); self.ht.set_text(f'h={h.height_m:.1f}m'); self.vt.set_text(f'v={vmag:.2f}m/s'); self.dt.set_text(f'dist={d:.2f}m'); self.st.set_text(f'status={status}')
        return self.trail,self.rock,self.tt,self.ht,self.vt,self.dt,self.ct,self.st

    def show(self): plt.show()
