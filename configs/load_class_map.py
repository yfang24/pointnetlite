import json
from pathlib import Path

def load_class_map(path):
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)