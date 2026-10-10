import csv
import os
from typing import List


def history_to_csv(history: List, path: str) -> str:
    if not history:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t"])
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h0 = history[0]
    # collect available fields
    rows = []
    header = ["t", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "acc_x", "acc_y", "acc_z",
              "height_m", "mass_kg", "fuel_kg", "dry_mass_kg", "q_w", "q_x", "q_y", "q_z",
              "omega_x", "omega_y", "omega_z", "alpha_x", "alpha_y", "alpha_z",
              "f_x", "f_y", "f_z", "m_x", "m_y", "m_z"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for h in history:
            writer.writerow([
                h.t,
                h.pos[0], h.pos[1], h.pos[2],
                h.vel[0], h.vel[1], h.vel[2],
                h.acc[0], h.acc[1], h.acc[2],
                getattr(h, "height_m", h.pos[2]),
                getattr(h, "mass_kg", 0.0),
                getattr(h, "fuel_kg", 0.0),
                getattr(h, "dry_mass_kg", 0.0),
                h.q[0], h.q[1], h.q[2], h.q[3],
                h.omega[0], h.omega[1], h.omega[2],
                h.alpha[0], h.alpha[1], h.alpha[2],
                h.force_total[0], h.force_total[1], h.force_total[2],
                h.moment_total[0], h.moment_total[1], h.moment_total[2],
            ])
    return path
