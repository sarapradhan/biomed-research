from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration and create output directories."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = Path(cfg["paths"]["output_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    logs_dir = Path(cfg["paths"]["logs_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
