# selflanding_rocket

Modular 3D rocket simulation focused on:
- Physically traceable dynamics (translation + rotation)
- Sensor models with uncertainty (noise, bias, drift, delay, dropout)
- State estimation with uncertainty awareness
- Transparent system monitoring & failure logging
- Reproducibility, validation, and systematic fault injection

## Quickstart

Run console simulation (generates CSV + plots in `output/`):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 main.py
```

Run 3D visual simulation (interactive, uses real sim data):

```bash
cd selflanding_rocket
PYTHONPATH=. python3 visual_app.py
```

Run tests:

```bash
cd selflanding_rocket
PYTHONPATH=. python3 -m pytest -v
```

## 3D Visual (matplotlib)

- 3D scene: ground plane, axes (X=red, Y=blue, Z=green), rocket marker, flight trail
- Uses actual State history from Python simulation (no made-up trajectory)
- Controls: Pause/Play, Restart
- HUD: t [s], h [m], v [m/s]
- Notes: Rotation stored but not integrated in current dynamics; trajectory = true translation. No autoland.

## Output files (main.py)

- `output/trajectory.csv` – true simulation state
- `output/summary.json` – basic summary
- `output/height.png`, `output/velocity.png`

View plots: `open output/height.png` (macOS) or open in Finder.

## Architecture
(see existing structure; visualization now includes `viz3d_mpl.py` + `visual_app.py`)

## Landing 3D Visual (Lernversion)

```bash
cd selflanding_rocket
PYTHONPATH=. python3 landing_app.py
```

Features: Zielpunkt (X,Y) per Slider änderbar, Start/Pause/Neustart, Anzeige t/h/v/dist(Ziel). Nutzt echte Simulationsdaten. Steuerung/Landung nicht implementiert (klar gekennzeichnet).
