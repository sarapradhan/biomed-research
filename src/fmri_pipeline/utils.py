from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


def setup_logging(log_dir: str, name: str = "pipeline") -> logging.Logger:
    """Configure file and console logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds where possible."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: Dict, path: str) -> None:
    """Save dictionary as JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_basename(run_row: Dict) -> str:
    """Create stable run key from BIDS entities."""
    sub = run_row.get("subject")
    ses = run_row.get("session")
    task = run_row.get("task")
    run = run_row.get("run")
    parts = [f"sub-{sub}"]
    if ses:
        parts.append(f"ses-{ses}")
    if task:
        parts.append(f"task-{task}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts)


def metric_path(output_root: str, metric: str, subject: str, run_key: Optional[str] = None) -> Path:
    """Return a standardized metric output path."""
    p = Path(output_root) / metric / f"sub-{subject}"
    if run_key:
        p = p / run_key
    p.mkdir(parents=True, exist_ok=True)
    return p


def upper_triangle_vector(matrix: np.ndarray, k: int = 1) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Flatten upper-triangle entries of a square matrix."""
    idx = np.triu_indices(matrix.shape[0], k=k)
    return matrix[idx], idx


def rebuild_symmetric(vec: np.ndarray, idx: Tuple[np.ndarray, np.ndarray], n: int) -> np.ndarray:
    """Reconstruct symmetric matrix from upper-triangle vector."""
    mat = np.zeros((n, n), dtype=float)
    mat[idx] = vec
    mat[(idx[1], idx[0])] = vec
    np.fill_diagonal(mat, 0.0)
    return mat
