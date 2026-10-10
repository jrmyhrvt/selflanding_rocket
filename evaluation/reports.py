import json
from datetime import datetime

class Report:
    def write_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def summarize(history, path=None):
    if not history:
        data = {"count": 0}
    else:
        h0, hf = history[0], history[-1]
        data = {
            "count": len(history),
            "t0": h0.t,
            "tf": hf.t,
            "dt": (hf.t - h0.t) / (len(history) - 1) if len(history) > 1 else 0.0,
            "pos0": [float(x) for x in h0.pos],
            "posf": [float(x) for x in hf.pos],
            "v0": [float(x) for x in h0.vel],
            "vf": [float(x) for x in hf.vel],
            "height0": float(getattr(h0, "height_m", h0.pos[2])),
            "heightf": float(getattr(hf, "height_m", hf.pos[2])),
            "mass0": float(h0.mass_kg),
            "massf": float(hf.mass_kg),
        }
    if path:
        Report().write_json(path, data)
    return data
