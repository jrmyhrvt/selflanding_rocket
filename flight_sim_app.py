"""Launch the unified 3D flight view (learning simulation).

Usage:
    PYTHONPATH=. python3 flight_sim_app.py

Controls:
    Start / Pause / Neustart / dt   - run the ONE shared simulation
    Aufrecht / Geneigt              - initial attitude only (same physics)
    Gesamtansicht / Nahansicht      - whole trajectory vs. rocket close-up
    Auto (Regler) / Manuell (Servo) - automatic controller vs. manual servos
    Servo X / Servo Y sliders       - MANUAL servo setpoints (deg); moving a
                                      slider recomputes the run and says so
                                      on screen (no silent reset)
    3D view: left-drag = rotate, right/middle-drag = pan, wheel = zoom
             keys: '+'/'-' zoom, 'o' overview, 'c' close-up, 'r' reset

The two virtual TVC servos turn a commanded setpoint into a rate-limited ACTUAL
nozzle angle; only the actual angle drives the thrust force and moment.
"""
from visualization.flight_view import FlightView


def main():
    view = FlightView()
    view.show()


if __name__ == "__main__":
    main()
