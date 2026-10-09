import numpy as np

class MassModel:
    def __init__(self, dry_mass_kg: float = 500.0, fuel_kg: float = 100.0):
        self.dry_mass_kg = dry_mass_kg
        self.fuel_kg = max(0.0, fuel_kg)
        self._est_fuel_kg = self.fuel_kg

    def total_mass(self) -> float:
        return self.dry_mass_kg + self.fuel_kg

    def burn(self, rate_kg_s: float, dt: float):
        burned = max(0.0, rate_kg_s * dt)
        self.fuel_kg = max(0.0, self.fuel_kg - burned)
        return burned

    def set_est_fuel(self, val_kg: float):
        self._est_fuel_kg = max(0.0, val_kg)

    def est_total_mass(self) -> float:
        return self.dry_mass_kg + self._est_fuel_kg
