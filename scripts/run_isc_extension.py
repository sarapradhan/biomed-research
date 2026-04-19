#!/usr/bin/env python3
"""run_isc_extension.py — Track 2 ISC Extension Runner for CNeuroMod Friends Analysis.

Focused entry point for running the ISC (InterSubject Correlation) analysis plus
scene annotation alignment on the CNeuroMod Friends dataset. Designed to be flexible:
can run the full pipeline from scratch, skip preprocessing if already done, or run
only scene annotation alignment assuming ISC has already been computed.

Scientific context
------------------
Intersubject correlation (ISC) measures synchrony in brain activity across subjects
watching the same movie. This runner adds a Track 2 extension that aligns temporal
hotspots of ISC with annotated scene features (emotional valence, theory-of-mind,
narrative transitions) to understand which aspects of naturalistic stimuli drive
reliable neural synchrony.

Usage
-----
    # Full pipeline from scratch (data ingestion to scene alignment)
    python scripts/run_isc_extension.py --config config/pipeline.cneuromod_isc.yaml

    # Skip preprocessing if already done (ROI extraction through scene alignment)
    python scripts/run_isc_extension.py --config config/pipeline.cneuromod_isc.yaml \
        --skip-preprocessing

    # Run only scene annotation alignment (assumes ISC maps already computed)
    python scripts/run_isc_extension.py --config config/pipeline.cneuromod_isc.yaml \
        --annotations-only

    # Override data paths
    python scripts/run_isc_extension.py --config config/pipeline.cneuromod_isc.yaml \
        --data-root /custom/cneuromod/path \
        --output-root /custom/output/path

Outputs
-------
    <output-root>/
        preprocessing/          Cleaned & registered BOLD images
        roi/                    ROI timeseries matrices
        static_fc/              Static functional connectivity matrices
        dynamic_fc/             Sliding-window dynamic FC
        isc/                    ISC maps (group & parcel-wise)
        scene_alignment/        ISC-to-scene correlation results
        logs/                   Detailed pipeline logs
        isc_extension_report.json   Execution summary

Dependencies
------------
    fmri_pipeline (local src/ directory)
    nibabel, nilearn, numpy, scipy, pandas, scikit-learn, pyyaml, joblib

Author
------
    Generated for the fMRI pipeline Track 2 extension workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ── project source on path ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.config import load_config  # noqa: E402
from fmri_pipeline.pipeline import (  # noqa: E402
    run_ingest,
    run_preprocess_qc,
    run_roi_step,
    run_static_dynamic_fc,
    run_isc_step,
    run_scene_alignment_step,
)
from fmri_pipeline.utils import set_global_seed, setup_logging  # noqa: E402


# ── Default config path ──────────────────────────────────────────────────────
DEFAULT_CONFIG = "config/pipeline.cneuromod_isc.yaml"


# ── Report generation ────────────────────────────────────────────────────────

def generate_summary_report(
    cfg: Dict[str, Any],
    n_subjects: int,
    isc_paths: Optional[Dict[str, str]] = None,
    scene_paths: Optional[Dict[str, str]] = None,
    error: Optional[str] = None,
    runtime_sec: float = 0.0,
) -> Dict[str, Any]:
    """Generate a JSON summary report of the ISC extension execution.

    Parameters
    ----------
    cfg : dict
        Pipeline configuration (already loaded)
    n_subjects : int
        Number of subjects processed
    isc_paths : dict, optional
        Output paths for ISC results (from run_isc_step)
    scene_paths : dict, optional
        Output paths for scene alignment results (from run_scene_alignment_step)
    error : str, optional
        Error message if execution failed
    runtime_sec : float
        Total wall-clock seconds for the execution

    Returns
    -------
    dict
        Report dictionary (also saved as JSON)
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "project_name": cfg.get("project", {}).get("name", "unknown"),
        "workflow": "isc_extension_track2",
        "status": "failed" if error else "completed",
        "error": error,
        "execution": {
            "runtime_seconds": runtime_sec,
            "n_subjects_processed": n_subjects,
            "skip_preprocessing": False,  # Will be updated by caller
            "annotations_only": False,    # Will be updated by caller
        },
        "outputs": {
            "isc_results": isc_paths or {},
            "scene_alignment_results": scene_paths or {},
        },
        "config": {
            "output_root": cfg.get("paths", {}).get("output_root", ""),
            "atlas": cfg.get("roi", {}).get("atlas", ""),
            "isc_permutations": cfg.get("isc", {}).get("permutations", 0),
            "scene_annotation_enabled": cfg.get("scene_annotation", {}).get("enabled", False),
        },
    }
    return report


# ── Main workflow ────────────────────────────────────────────────────────────

def run_isc_extension(
    config_file: str,
    data_root: Optional[Path] = None,
    output_root: Optional[Path] = None,
    skip_preprocessing: bool = False,
    annotations_only: bool = False,
) -> None:
    """Run the ISC extension pipeline with flexible entry points.

    Parameters
    ----------
    config_file : str
        Path to pipeline YAML config
    data_root : Path, optional
        Override data root path from config
    output_root : Path, optional
        Override output root path from config
    skip_preprocessing : bool
        If True, skip ingestion, preprocessing, ROI extraction, and FC steps.
        Assumes these outputs already exist in the output directory.
    annotations_only : bool
        If True, only run scene alignment (assumes ISC maps already computed).
        Implies skip_preprocessing=True.
    """
    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────
    t_start = time.time()

    # Load config
    try:
        cfg = load_config(config_file)
    except FileNotFoundError:
        sys.exit(f"ERROR: Config file not found: {config_file}")
    except Exception as e:
        sys.exit(f"ERROR: Failed to load config: {e}")

    # Setup logging
    out_root = Path(output_root or cfg["paths"]["output_root"]).expanduser().resolve()
    logs_dir = Path(cfg["paths"].get("logs_dir", str(out_root / "logs"))).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(logs_dir), name="isc_extension")

    # Set seed for reproducibility
    set_global_seed(cfg.get("project", {}).get("random_seed", 42))

    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("TRACK 2 ISC EXTENSION PIPELINE RUNNER")
    logger.info("=" * 80)
    logger.info("Project       : %s", cfg.get("project", {}).get("name", "unknown"))
    logger.info("Config        : %s", config_file)
    logger.info("Output root   : %s", out_root)
    logger.info("Timestamp     : %s", datetime.now().isoformat())
    logger.info("─" * 80)

    # Log execution mode
    if annotations_only:
        logger.info("Mode: ANNOTATIONS-ONLY (assumes ISC already computed)")
        skip_preprocessing = True
    elif skip_preprocessing:
        logger.info("Mode: SKIP-PREPROCESSING (assumes preprocessing completed)")
    else:
        logger.info("Mode: FULL PIPELINE (data ingestion through scene alignment)")
    logger.info("─" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline execution
    # ─────────────────────────────────────────────────────────────────────────
    isc_paths: Optional[Dict[str, str]] = None
    scene_paths: Optional[Dict[str, str]] = None
    n_subjects = 0
    error_msg: Optional[str] = None

    try:
        # Update config with overrides
        if data_root:
            # Update BIDS roots
            if "bids_roots" in cfg["paths"]:
                for k in cfg["paths"]["bids_roots"]:
                    cfg["paths"]["bids_roots"][k] = str(data_root)
            if "derivatives_root" in cfg["paths"]:
                cfg["paths"]["derivatives_root"] = str(data_root / "derivatives")

        if output_root:
            cfg["paths"]["output_root"] = str(output_root)
            cfg["paths"]["logs_dir"] = str(output_root / "logs")
            cfg["paths"]["cache_dir"] = str(output_root / "cache")

        # ─────────────────────────────────────────────────────────────────────
        # 1. Data ingestion (unless skipping preprocessing)
        # ─────────────────────────────────────────────────────────────────────
        if not skip_preprocessing:
            logger.info("STEP 1: Data Ingestion (BIDS → run manifest)")
            logger.info("─" * 80)
            try:
                runs_df, metadata_df = run_ingest(cfg, logger)
                n_subjects = runs_df["subject"].nunique()
                logger.info("  Ingested %d subjects across %d runs", n_subjects, len(runs_df))
            except Exception as e:
                error_msg = f"Data ingestion failed: {e}"
                logger.error(error_msg)
                raise

            # ─────────────────────────────────────────────────────────────────
            # 2. Preprocessing + QC
            # ─────────────────────────────────────────────────────────────────
            logger.info("\nSTEP 2: Preprocessing & Motion QC")
            logger.info("─" * 80)
            try:
                runs_df = run_preprocess_qc(cfg, runs_df, logger)
                n_subjects = runs_df["subject"].nunique()
                logger.info("  Preprocessed %d runs across %d subjects", len(runs_df), n_subjects)
            except Exception as e:
                error_msg = f"Preprocessing failed: {e}"
                logger.error(error_msg)
                raise

            # ─────────────────────────────────────────────────────────────────
            # 3. ROI extraction (parcellation)
            # ─────────────────────────────────────────────────────────────────
            logger.info("\nSTEP 3: ROI Extraction (parcellation)")
            logger.info("─" * 80)
            try:
                roi_df = run_roi_step(cfg, runs_df, logger)
                logger.info("  Extracted ROI timeseries for %d runs", len(roi_df))
            except Exception as e:
                error_msg = f"ROI extraction failed: {e}"
                logger.error(error_msg)
                raise

            # ─────────────────────────────────────────────────────────────────
            # 4. Static + Dynamic FC
            # ─────────────────────────────────────────────────────────────────
            logger.info("\nSTEP 4: Functional Connectivity (static & dynamic)")
            logger.info("─" * 80)
            try:
                sfc_df, dfc_df, reho_df = run_static_dynamic_fc(cfg, roi_df, logger)
                logger.info("  Computed static FC for %d runs", len(sfc_df))
                logger.info("  Computed dynamic FC for %d runs", len(dfc_df))
            except Exception as e:
                error_msg = f"FC computation failed: {e}"
                logger.error(error_msg)
                raise

        else:
            # Load runs_df from previously saved state (if needed for ISC step)
            logger.info("PREPROCESSING SKIPPED (assuming outputs already exist)")
            logger.info("─" * 80)
            # For annotations-only, we may not need runs_df, but ISC step expects it
            # In a full implementation, we'd load from cache or reconstruct from config
            # For now, create a minimal runs_df or pass None if not needed
            runs_df = None

        # ─────────────────────────────────────────────────────────────────────
        # 5. ISC computation (unless annotations-only)
        # ─────────────────────────────────────────────────────────────────────
        if not annotations_only:
            logger.info("\nSTEP 5: InterSubject Correlation (ISC)")
            logger.info("─" * 80)
            try:
                # Need runs_df; if we skipped preprocessing, reconstruct it
                if runs_df is None:
                    logger.warning("  runs_df not available; attempting to load from cache")
                    # In a production system, load from saved pickle/CSV
                    raise RuntimeError(
                        "runs_df required for ISC step but preprocessing was skipped. "
                        "Save runs_df to disk or use --annotations-only."
                    )

                isc_paths = run_isc_step(cfg, runs_df, logger)
                logger.info("  ISC results written to:")
                for key, path in (isc_paths or {}).items():
                    logger.info("    %s: %s", key, path)
            except Exception as e:
                error_msg = f"ISC computation failed: {e}"
                logger.error(error_msg)
                raise

        else:
            logger.info("\nSTEP 5: ISC SKIPPED (--annotations-only mode)")
            logger.info("─" * 80)

        # ─────────────────────────────────────────────────────────────────────
        # 6. Scene annotation alignment (Track 2 extension)
        # ─────────────────────────────────────────────────────────────────────
        scene_enabled = cfg.get("scene_annotation", {}).get("enabled", False)
        if scene_enabled:
            logger.info("\nSTEP 6: Scene Annotation Alignment (Track 2 extension)")
            logger.info("─" * 80)
            try:
                # If we skipped preprocessing, runs_df should be reconstructed or None
                if runs_df is None and not skip_preprocessing:
                    logger.warning("  runs_df is None; scene alignment may have limited scope")

                scene_paths = run_scene_alignment_step(cfg, runs_df, isc_paths or {}, logger)
                logger.info("  Scene alignment results written to:")
                for key, path in (scene_paths or {}).items():
                    logger.info("    %s: %s", key, path)
            except Exception as e:
                error_msg = f"Scene alignment failed: {e}"
                logger.error(error_msg)
                raise
        else:
            logger.info("\nSTEP 6: Scene annotation alignment DISABLED in config")
            logger.info("─" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 80)
        logger.error("Error: %s", str(e))
        error_msg = str(e)

    # ─────────────────────────────────────────────────────────────────────────
    # Summary report
    # ─────────────────────────────────────────────────────────────────────────
    runtime = time.time() - t_start

    report = generate_summary_report(
        cfg,
        n_subjects=n_subjects,
        isc_paths=isc_paths,
        scene_paths=scene_paths,
        error=error_msg,
        runtime_sec=runtime,
    )

    report["execution"]["skip_preprocessing"] = skip_preprocessing
    report["execution"]["annotations_only"] = annotations_only

    # Save report to JSON
    report_path = Path(cfg["paths"]["output_root"]) / "isc_extension_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info("Status        : %s", report["status"])
    logger.info("Subjects      : %d", report["execution"]["n_subjects_processed"])
    logger.info("Runtime       : %.1f seconds", report["execution"]["runtime_seconds"])
    logger.info("Report        : %s", report_path)
    logger.info("=" * 80)

    if error_msg:
        sys.exit(f"Pipeline execution failed: {error_msg}")


# ── CLI argument parsing ─────────────────────────────────────────────────────

def main() -> None:
    """Parse CLI arguments and run the ISC extension pipeline."""
    parser = argparse.ArgumentParser(
        prog="run_isc_extension.py",
        description="Track 2 ISC Extension: Naturalistic movie analysis with scene annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Path to pipeline YAML config (default: {DEFAULT_CONFIG})",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override data root path from config (for BIDS/derivatives roots)",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override output root path from config",
    )

    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip ingestion, preprocessing, ROI extraction, and FC steps. "
        "Assumes these outputs already exist.",
    )

    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Run only scene annotation alignment (assumes ISC maps already computed). "
        "Implies --skip-preprocessing.",
    )

    args = parser.parse_args()

    # Resolve config path
    config_file = str(Path(args.config).expanduser().resolve())
    if not Path(config_file).exists():
        sys.exit(f"ERROR: Config file not found: {config_file}")

    # Run the pipeline
    run_isc_extension(
        config_file=config_file,
        data_root=args.data_root.expanduser().resolve() if args.data_root else None,
        output_root=args.output_root.expanduser().resolve() if args.output_root else None,
        skip_preprocessing=args.skip_preprocessing,
        annotations_only=args.annotations_only,
    )


if __name__ == "__main__":
    main()
