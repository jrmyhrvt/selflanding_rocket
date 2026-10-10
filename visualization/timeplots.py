import os
import matplotlib.pyplot as plt
from typing import List


def plot_height(history: List, path: str = "data/height.png"):
    if not history:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = [h.t for h in history]
    z = [getattr(h, "height_m", h.pos[2]) for h in history]
    plt.figure()
    plt.plot(t, z, label="height")
    plt.xlabel("t [s]")
    plt.ylabel("height [m]")
    plt.title("Height over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_velocity(history: List, path: str = "data/velocity.png"):
    if not history:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = [h.t for h in history]
    vx = [h.vel[0] for h in history]
    vy = [h.vel[1] for h in history]
    vz = [h.vel[2] for h in history]
    plt.figure()
    plt.plot(t, vx, label="vx")
    plt.plot(t, vy, label="vy")
    plt.plot(t, vz, label="vz")
    plt.xlabel("t [s]")
    plt.ylabel("v [m/s]")
    plt.title("Velocity over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_position_components(history: List, path: str = "data/position.png"):
    if not history:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = [h.t for h in history]
    x = [h.pos[0] for h in history]
    y = [h.pos[1] for h in history]
    z = [h.pos[2] for h in history]
    plt.figure()
    plt.plot(t, x, label="x")
    plt.plot(t, y, label="y")
    plt.plot(t, z, label="z")
    plt.xlabel("t [s]")
    plt.ylabel("pos [m]")
    plt.title("Position over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
