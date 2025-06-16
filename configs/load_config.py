import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent

def load_config(config_name):
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"
        
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config, config_path
