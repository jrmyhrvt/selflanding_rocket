from dataclasses import dataclass, field
from typing import List
import time

@dataclass
class Event:
    t: float
    level: str  # info/warn/error
    msg: str

class Monitor:
    def __init__(self):
        self.events: List[Event] = []
        self.modes = {"state": "nominal"}

    def log(self, t: float, level: str, msg: str):
        self.events.append(Event(t=t, level=level, msg=msg))

    def check_plausible(self, state):
        if state.height_m < 0.0:
            self.log(state.t, "warn", "height_m < 0")
        if state.mass_kg < 0.0:
            self.log(state.t, "error", "negative mass")
