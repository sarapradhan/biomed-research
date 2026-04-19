"""YAML configuration loader and output directory initializer.

Reads the pipeline's single-source-of-truth configuration file and ensures
that all required output, cache, and log directories exist before any
analysis stage executes. Also validates optional configuration blocks
(e.g., scene_annotation) for correctness.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml


def validate_scene_annotation_config(scene_cfg: Dict[str, Any], logger: logging.Logger) -> None:
    """Validate scene_annotation configuration block if enabled.
    
    Parameters
    ----------
    scene_cfg : dict
        The scene_annotation configuration dictionary.
    logger : logging.Logger
        Logger for reporting validation issues.
    
    Raises
    ------
    ValueError
        If enabled but required fields are missing or invalid.
    """
    if not scene_cfg.get("enabled", False):
        logger.debug("scene_annotation disabled, skipping validation")
        return

    # Check required fields
    required_fields = ["annotation_dir", "tr_sec", "features"]
    for field in required_fields:
        if field not in scene_cfg:
            raise ValueError(
                f"scene_annotation.enabled=true but required field '{field}' is missing. "
                f"See docs/SCENE_ANNOTATION_SCHEMA.md for details."
            )

    # Validate tr_sec
    try:
        tr_sec = float(scene_cfg["tr_sec"])
        if tr_sec <= 0:
            raise ValueError("tr_sec must be positive")
    except (TypeError, ValueError) as e:
        raise ValueError(f"scene_annotation.tr_sec must be a positive float, got {scene_cfg['tr_sec']}") from e

    # Validate features list
    if not isinstance(scene_cfg["features"], list) or len(scene_cfg["features"]) == 0:
        raise ValueError("scene_annotation.features must be a non-empty list of strings")
    if not all(isinstance(f, str) for f in scene_cfg["features"]):
        raise ValueError("scene_annotation.features must contain only strings")

    # Validate alignment subblock if present
    alignment = scene_cfg.get("alignment", {})
    if alignment:
        if not isinstance(alignment, dict):
            raise ValueError("scene_annotation.alignment must be a dictionary")
        
        # Validate summary_method
        if "summary_method" in alignment:
            if alignment["summary_method"] not in ["mean", "median"]:
                raise ValueError(
                    f"scene_annotation.alignment.summary_method must be 'mean' or 'median', "
                    f"got '{alignment['summary_method']}'"
                )
        
        # Validate min_scene_trs
        if "min_scene_trs" in alignment:
            try:
                min_trs = int(alignment["min_scene_trs"])
                if min_trs <= 0:
                    raise ValueError("min_scene_trs must be positive")
            except (TypeError, ValueError) as e:
                raise ValueError(f"scene_annotation.alignment.min_scene_trs must be positive int") from e
        
        # Validate zscore_isc
        if "zscore_isc" in alignment:
            if not isinstance(alignment["zscore_isc"], bool):
                raise ValueError("scene_annotation.alignment.zscore_isc must be true or false")

    # Validate correlation_method if present
    if "correlation_method" in scene_cfg:
        if scene_cfg["correlation_method"] not in ["pearson", "spearman"]:
            raise ValueError(
                f"scene_annotation.correlation_method must be 'pearson' or 'spearman', "
                f"got '{scene_cfg['correlation_method']}'"
            )

    logger.info("scene_annotation config validated successfully")


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration, validate optional blocks, and create output directories.
    
    Parameters
    ----------
    path : str
        Path to YAML configuration file.
    
    Returns
    -------
    dict
        Loaded and validated configuration dictionary.
    
    Raises
    ------
    ValueError
        If configuration is invalid.
    FileNotFoundError
        If configuration file does not exist.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Set up logger for validation
    logger = logging.getLogger(__name__)
    
    # Validate optional configuration blocks
    if "scene_annotation" in cfg:
        validate_scene_annotation_config(cfg["scene_annotation"], logger)

    out_root = Path(cfg["paths"]["output_root"])
    cache_dir = Path(cfg["paths"]["cache_dir"])
    logs_dir = Path(cfg["paths"]["logs_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
