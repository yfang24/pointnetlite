import json
from pathlib import Path

# def load_class_map(path):
#     path = Path(path)
#     with open(path, "r") as f:
#         return json.load(f)

def load_class_map(name_or_path: str):
    """Load a class map by short name (e.g., 'modelnet11') or direct JSON path."""
    path = Path(name_or_path)
    if not path.suffix: # no extension, e.g., 'modelnet11'
        path = Path(__file__).parent / f"{name_or_path}.json"
    elif not path.is_absolute():    # relative path, e.g., 'modelnet11.json'
        # resolve relative paths too
        path = Path(__file__).parent / path

    if not path.exists():
        raise FileNotFoundError(f"Class map not found: {path}")

    with open(path, "r") as f:
        return json.load(f)