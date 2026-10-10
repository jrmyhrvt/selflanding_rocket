# Visualization & Export

## Export (CSV)
- `evaluation.export.history_to_csv(history, path)` exports all available true-state fields from `State`:
  - t, pos_x/y/z, vel_x/y/z, acc_x/y/z, height_m, mass_kg/fuel_kg/dry_mass_kg
  - q_w/q_x/q_y/q_z, omega_x/y/z, alpha_x/y/z, f_x/y/z, m_x/y/z
- Only existing values are written (no fabricated sensor/estimate values).

## Plots (time series)
- `visualization.timeplots.plot_height(history, path)` 
- `visualization.timeplots.plot_velocity(history, path)`
- `visualization.timeplots.plot_position_components(history, path)`
- matplotlib >= 3.7 recommended; system has 3.9.4.

## 3D Animation
- Not implemented. Rotation is present in state but not integrated by current dynamics; only translation+gravity active. See `visualization/animation.py`.

## Quick run
```bash
cd selflanding_rocket
PYTHONPATH=. python3 run_analysis.py --tf 2.0 --dt 0.01
```
Outputs: data/trajectory.csv, data/summary.json, data/height.png, data/velocity.png, data/position.png
