"""
Simple 2D/3D animation helpers are NOT implemented by default because
the current simulation state is sufficient for time series, but a
full interactive 3D animation requires additional dependencies and
careful handling of rotation. 

Current state supports: t, pos, vel, acc, q, omega, alpha, mass/fuel, height.

Constraints/notes:
- Only translation + gravity active in loop (rotation not integrated).
- q exists but not updated by dynamics here.
- 3D view would be easy to add later with matplotlib's FuncAnimation
  or plotly, but we avoid inventing a fake 6DOF visualization now.

If you want a minimal 3D trajectory plot (static), use visualization.plotter/plot3d
later. For now, prefer timeplots.
"""
__all__ = ["summary"]
def summary():
    return "3D animation not implemented: rotation/dynamics coupling minimal; use timeplots + CSV."
