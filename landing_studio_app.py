#!/usr/bin/env python3
"""
Interactive Autoland Learning Studio (simplified virtual rocket landing).

Run:
    PYTHONPATH=. python3 landing_studio_app.py

This is a virtual learning tool only. No hardware, no real-rocket control.
"""
from simulation.scenario import Scenario
from visualization.viz3d_studio import LandingStudio


def main():
    studio = LandingStudio(Scenario())
    print("Autoland Lern-Studio gestartet.")
    print("Links: 3D-Flugbahn. Rechts: Hoehe / Geschwindigkeit / Zielabstand.")
    print("Unten: Szenario-Einstellungen und Buttons "
          "(Start, Pause, Reset, Neu berechnen).")
    print("Regler/Buttons sind interaktiv. Fenster schliessen zum Beenden.")
    studio.show()


if __name__ == "__main__":
    main()
