# Visuelle Simulation starten

## Voraussetzungen
- Python 3.9+
- Pakete: numpy, matplotlib (bereits installiert)

## Starten
```bash
cd selflanding_rocket
PYTHONPATH=. python3 visual_app.py
```

## Bedienung
- Pause/Play: pausiert/fortsetzt Animation
- Restart: setzt Animation auf Start zurück
- HUD zeigt t [s], h [m], v [m/s]
- 3D-Szene mit Boden, Koordinatenachsen, Raketenmarker und Flugbahn

## Hinweise
- Verwendet echte Simulationsdaten aus `simulation.loop.Simulator.run()` (keine erfundenen Werte)
- Rotation im Zustand vorhanden, aber nicht aktiv integriert
- Keine aktive Steuerung/Landung
