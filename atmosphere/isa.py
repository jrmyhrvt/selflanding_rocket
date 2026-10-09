import numpy as np
from config import units

def density(h_m: float):
    if h_m < 0.0:
        h_m = 0.0
    T = units.SEA_LEVEL_TEMP_K + units.LAPSE_RATE_K_M * h_m
    if T <= 0.0:
        T = units.SEA_LEVEL_TEMP_K
    p = units.SEA_LEVEL_PRESSURE_PA * (T / units.SEA_LEVEL_TEMP_K) ** (units.GRAVITY_M_S2 / (-units.LAPSE_RATE_K_M * units.GAS_CONSTANT))
    rho = p / (units.GAS_CONSTANT * T)
    return rho

def pressure(h_m: float):
    if h_m < 0.0:
        h_m = 0.0
    T = units.SEA_LEVEL_TEMP_K + units.LAPSE_RATE_K_M * h_m
    if T <= 0.0:
        T = units.SEA_LEVEL_TEMP_K
    p = units.SEA_LEVEL_PRESSURE_PA * (T / units.SEA_LEVEL_TEMP_K) ** (units.GRAVITY_M_S2 / (-units.LAPSE_RATE_K_M * units.GAS_CONSTANT))
    return p

def temperature(h_m: float):
    if h_m < 0.0:
        h_m = 0.0
    T = units.SEA_LEVEL_TEMP_K + units.LAPSE_RATE_K_M * h_m
    if T <= 0.0:
        T = 0.0001
    return T

def speed_of_sound(h_m: float):
    T = temperature(h_m)
    return np.sqrt(1.4 * units.GAS_CONSTANT * T)
