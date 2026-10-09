import json
from datetime import datetime

class Report:
    def write_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
