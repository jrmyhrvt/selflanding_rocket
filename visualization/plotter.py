import matplotlib.pyplot as plt
import numpy as np

def plot_pos(history, path="data/pos.png"):
    t = [h.t for h in history]
    x = [h.pos[0] for h in history]
    y = [h.pos[1] for h in history]
    z = [h.pos[2] for h in history]
    plt.figure()
    plt.plot(t, x, label="x")
    plt.plot(t, y, label="y")
    plt.plot(t, z, label="z")
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel("pos [m]")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# keep existing; also expose helpers
from .timeplots import plot_height, plot_velocity, plot_position_components
