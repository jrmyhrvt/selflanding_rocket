"""Launch the unified 3D flight view (learning simulation).

Usage:
    PYTHONPATH=. python3 flight_sim_app.py
"""
from visualization.flight_view import FlightView


def main():
    view = FlightView()
    view.show()


if __name__ == "__main__":
    main()
