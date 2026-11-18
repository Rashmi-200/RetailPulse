from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
