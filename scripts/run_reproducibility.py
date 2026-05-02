#!/usr/bin/env python3
"""End-to-end reproducibility harness (JEI revision Phase 1 + 2 + 3).

Walks ``execution.run_order`` from ``config/reproducibility.yaml`` and
dispatches each named step to its module's ``run(config)`` entry point.
Modules that are absent from ``execution.run_order`` or whose section's
``enabled: false`` is set are skipped.

A ``manifest.json`` is written to ``paths.output_root`` capturing the
git SHA, package versions, config hash, per-step status, and timings so
the run is auditable after the fact.

Example:

    python scripts/run_reproducibility.py --config config/reproducibility.yaml
    python scripts/run_reproducibility.py --config config/reproducibility.yaml \\
        --only fc,reho               # restrict to a subset
    python scripts/run_reproducibility.py --config config/reproducibility.yaml \\
        --skip ica,graph             # exclude listed steps
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Map of step name -> dotted module path implementing run(config).
STEP_MODULES: Dict[str, str] = {
    "fc": "fmri_pipeline.reproducibility.fc_reproducibility",
    "reho": "fmri_pipeline.reproducibility.reho_stability",
    "ica": "fmri_pipeline.reproducibility.ica_stability",
    "graph": "fmri_pipeline.reproducibility.graph_stability",
    "dfc": "fmri_pipeline.reproducibility.dfc_sensitivity",
    "network_anchor": "fmri_pipeline.reproducibility.network_anchor",
    "scorecard": "fmri_pipeline.reproducibility.scorecard",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_reproducibility")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _config_hash(config: Dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": platform.python_version()}
    for pkg in ("numpy", "scipy", "pandas", "nibabel", "nilearn",
                "scikit-learn", "statsmodels", "pybids", "joblib", "networkx"):
        try:
            mod = importlib.import_module(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def _is_step_enabled(step: str, config: Dict[str, Any]) -> bool:
    """Return True unless the step's section explicitly sets ``enabled: false``.

    If the step is named in ``execution.run_order`` it runs by default,
    even when its config section is absent. The two nested-section steps
    (``ica.stability`` and ``graph.stability``) honour their own enabled
    flag for backward compatibility with the existing config schema.
    """
    if step == "ica":
        return bool(((config.get("ica") or {}).get("stability") or {}).get("enabled", True))
    if step == "graph":
        return bool(((config.get("graph") or {}).get("stability") or {}).get("enabled", True))
    section = config.get(step)
    if isinstance(section, dict) and "enabled" in section:
        return bool(section["enabled"])
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--only",
        help="Comma-separated subset of steps to run (overrides execution.run_order).",
    )
    parser.add_argument(
        "--skip",
        help="Comma-separated steps to skip.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved step plan and exit without executing.",
    )
    args = parser.parse_args(argv)

    logger = _setup_logger()

    with args.config.open() as f:
        config = yaml.safe_load(f)

    execution = config.get("execution") or {}
    run_order: List[str] = list(execution.get("run_order") or list(STEP_MODULES.keys()))
    continue_on_error = bool(execution.get("continue_on_error", True))
    write_manifest = bool(execution.get("write_manifest", True))

    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        run_order = [s for s in run_order if s in wanted]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",") if s.strip()}
        run_order = [s for s in run_order if s not in skip]

    unknown = [s for s in run_order if s not in STEP_MODULES]
    if unknown:
        logger.error("Unknown step(s) in run_order: %s. Known: %s",
                     unknown, sorted(STEP_MODULES))
        return 2

    plan = [(s, STEP_MODULES[s], _is_step_enabled(s, config)) for s in run_order]
    logger.info("Resolved run plan:")
    for step, mod, enabled in plan:
        logger.info("  %-15s -> %s%s", step, mod, "" if enabled else "  (disabled, will skip)")
    if args.dry_run:
        return 0

    output_root = Path(config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(args.config.resolve()),
        "config_hash": _config_hash(config),
        "git_sha": _git_sha(REPO_ROOT),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "steps": [],
    }

    overall_status = 0
    for step, module_path, enabled in plan:
        step_record: Dict[str, Any] = {
            "step": step,
            "module": module_path,
            "enabled": enabled,
            "status": "skipped" if not enabled else "pending",
            "error": None,
            "duration_seconds": 0.0,
        }
        if not enabled:
            logger.info("Skipping disabled step: %s", step)
            manifest["steps"].append(step_record)
            continue

        logger.info("Running step: %s", step)
        t0 = time.perf_counter()
        halt = False
        try:
            module = importlib.import_module(module_path)
            module.run(config)
            step_record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            step_record["status"] = "error"
            step_record["error"] = repr(exc)
            logger.error("Step %s failed: %s", step, exc)
            overall_status = 1
            halt = not continue_on_error
        finally:
            step_record["duration_seconds"] = round(time.perf_counter() - t0, 3)
            manifest["steps"].append(step_record)
            logger.info("  -> %s (%.2fs)", step_record["status"], step_record["duration_seconds"])
        if halt:
            break

    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["overall_status"] = "ok" if overall_status == 0 else "errors"

    if write_manifest:
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.info("Manifest written to %s", manifest_path)

    return overall_status


if __name__ == "__main__":
    sys.exit(main())
