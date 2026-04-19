#!/usr/bin/env python3
"""run_fmriprep_pilot.py — Standalone runner for the OpenNeuro ds007318 fMRIPrep pilot.

This script bypasses BIDSLayout/pybids so the pipeline can be run directly against
the fMRIPrep derivative structure on disk, without needing the BIDS root index.

Usage
-----
    python scripts/run_fmriprep_pilot.py [--data-root /path/to/fMRIPrep] [--output-root /path/to/output]

If --data-root is omitted the script defaults to
    ~/fmri-pipeline-data

Minimum requirements
---------------------
    pip install nibabel nilearn numpy scipy pandas scikit-learn statsmodels matplotlib pyyaml joblib pybids

Dataset
-------
    OpenNeuro ds007318 (working-memory removal task, Northwest Normal University)
    3 subjects with fMRIPrep derivatives (sub-01, sub-02, sub-03), 5 BOLD runs total
    Task label : removal  (resting-state equivalent for this dataset)
    TR         : 2.0 s
    Space      : MNI152NLin2009cAsym, res-2
    All subjects labelled group=patient; no control group → group_stats step skipped
    ISC        : skipped (requires dataset=="algonauts" and naturalistic movie data)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ── project source on path ────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.config import load_config  # noqa: E402
from fmri_pipeline.pipeline import (  # noqa: E402
    run_ica_step,
    run_pca_step,
    run_preprocess_qc,
    run_reho_step,
    run_roi_step,
    run_static_dynamic_fc,
)
from fmri_pipeline.roi import get_schaefer_atlas  # noqa: E402
from fmri_pipeline.utils import set_global_seed, setup_logging  # noqa: E402


# ── known runs ────────────────────────────────────────────────────────────────
# Each entry: (subject, session)  — task is always "removal", no run index
_KNOWN_RUNS: List[tuple] = [
    ("01", "1"),
    ("01", "2"),
    ("02", "1"),
    ("03", "1"),
    ("03", "2"),
]

_TR = 2.0
_SPACE = "MNI152NLin2009cAsym"
_RES = "2"
_TASK = "removal"
_DATASET = "schizconnect"
_DIAGNOSIS = "patient"  # all subjects in this dataset are labelled patient


def _bold_path(deriv: Path, sub: str, ses: str) -> Path:
    return (
        deriv
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "func"
        / f"sub-{sub}_ses-{ses}_task-{_TASK}_space-{_SPACE}_res-{_RES}_desc-preproc_bold.nii.gz"
    )


def _mask_path(deriv: Path, sub: str, ses: str) -> Path:
    return (
        deriv
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "func"
        / f"sub-{sub}_ses-{ses}_task-{_TASK}_space-{_SPACE}_res-{_RES}_desc-brain_mask.nii.gz"
    )


def _confounds_path(deriv: Path, sub: str, ses: str) -> Path:
    return (
        deriv
        / f"sub-{sub}"
        / f"ses-{ses}"
        / "func"
        / f"sub-{sub}_ses-{ses}_task-{_TASK}_desc-confounds_timeseries.tsv"
    )


def build_run_manifest(deriv: Path) -> pd.DataFrame:
    """Build a runs DataFrame directly from known derivative file paths.

    This replicates the output of bids_ingest.collect_runs() without needing
    a BIDS root index or pybids BIDSLayout scan.
    """
    rows = []
    for sub, ses in _KNOWN_RUNS:
        bold = _bold_path(deriv, sub, ses)
        mask = _mask_path(deriv, sub, ses)
        conf = _confounds_path(deriv, sub, ses)

        missing = [p for p in (bold, mask, conf) if not p.exists()]
        if missing:
            print(f"  [SKIP] sub-{sub} ses-{ses}: missing {[str(m) for m in missing]}")
            continue

        rows.append(
            {
                "dataset": _DATASET,
                "site": _DATASET,
                "subject": sub,
                "session": ses,
                "task": _TASK,
                "run": None,
                "bold_file": str(bold),
                "brain_mask_file": str(mask),
                "confounds_file": str(conf),
                "tr": _TR,
                "diagnosis": _DIAGNOSIS,
            }
        )

    if not rows:
        raise RuntimeError(
            f"No valid runs found under {deriv}.  "
            "Check that --data-root points to the fMRIPrep directory containing a 'derivatives/' sub-folder."
        )

    df = pd.DataFrame(rows)
    print(f"  Built manifest: {len(df)} runs across {df['subject'].nunique()} subjects.")
    return df


def build_cfg(data_root: Path, output_root: Path) -> Dict:
    """Build a pipeline config dict for the ds007318 pilot (no YAML file needed)."""
    logs_dir = output_root / "logs"
    cache_dir = output_root / "cache"
    return {
        "project": {
            "name": "ds007318_removal_pilot",
            "random_seed": 42,
            "n_jobs": 4,
            "debug_mode": False,
            "debug_subject_limit": None,
        },
        "paths": {
            "bids_roots": {_DATASET: str(data_root)},
            "derivatives_root": str(data_root / "derivatives"),
            "phenotypic_tsv": str(data_root / "phenotypic.tsv"),  # ok if absent
            "output_root": str(output_root),
            "cache_dir": str(cache_dir),
            "logs_dir": str(logs_dir),
        },
        "bids": {
            "space": _SPACE,
            "desc_preproc": "preproc",
            "use_aroma": False,
            "task_rest": _TASK,
            "task_movie": "__unused__",
        },
        "preprocessing": {
            "confounds": {
                "include_wm_csf": True,
                "wm_columns": ["white_matter", "wm"],
                "csf_columns": ["csf"],
                # GSR: the confounds TSV already contains a pre-computed global_signal column
                # from fMRIPrep. load_confound_matrix will pick it up via extract_global_signal
                # if include_global_signal is True (requires bold_file + mask_file).
                "include_global_signal": True,
            },
            "scrubbing": {
                "fd_threshold_mm": 0.5,
                "exclude_percent_censored": 0.20,
                "exclude_max_motion_mm": 3.0,
                "exclude_max_rotation_deg": 3.0,
            },
            "filtering": {
                "low_hz": 0.01,
                "high_hz": 0.10,
            },
            "smoothing_fwhm_mm": 6.0,
        },
        "roi": {
            "atlas": "schaefer_200",
            "schaefer_n_rois": 200,
            "schaefer_yeo_networks": 7,
            "standardize": "zscore_sample",
        },
        "reho": {
            "neighborhood": 26,
            "normalize_zscore": True,
            "chunk_size_voxels": 10000,
        },
        "dynamic_fc": {
            "window_trs": 30,
            "step_trs": 5,
            "exploratory_clustering": False,
            "exploratory_clusters": 4,
        },
        "ica": {
            "n_components": 20,
            "max_iter": 500,
            "corr_match_threshold": 0.2,
        },
        "pca": {
            "n_components": 5,
        },
        "isc": {
            "permutations": 1000,
            "min_subjects": 4,
            "use_circular_shift_null": True,
            "random_state": 42,
        },
        "stats": {
            # diagnosis_column set to a sentinel so group_stats silently skips:
            # all subjects are "patient"; there is no control group to contrast.
            "diagnosis_column": "__no_groups__",
            "patient_label": "patient",
            "control_label": "control",
            "covariates": ["mean_fd"],
            "optional_covariates_if_available": ["age", "sex", "site"],
            "fdr_q": 0.05,
        },
        "execution": {
            "run_order": [
                "preprocess_qc",
                "roi_timeseries",
                "reho",
                "static_fc",
                "dynamic_fc",
                "ica",
                "pca",
                # isc: skipped — requires dataset=="algonauts" (naturalistic movie)
                # group_stats: skipped — no control group in this dataset
            ]
        },
    }


def run_pilot(data_root: Path, output_root: Path) -> None:
    """Run the ds007318 pilot pipeline end-to-end."""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(data_root, output_root)
    logger = setup_logging(cfg["paths"]["logs_dir"], name="fmriprep_pilot")
    set_global_seed(int(cfg["project"]["random_seed"]))

    logger.info("=" * 70)
    logger.info("OpenNeuro ds007318 — removal-task pilot")
    logger.info("Data root   : %s", data_root)
    logger.info("Output root : %s", output_root)
    logger.info("=" * 70)

    # ── Step 0: build run manifest (bypasses BIDSLayout) ─────────────────────
    logger.info("Step 0 | Building run manifest from known derivative paths")
    deriv = data_root / "derivatives"
    runs_df = build_run_manifest(deriv)
    runs_df.to_csv(output_root / "manifests" / "run_manifest_raw.csv", index=False)

    # ── Step 1: preprocess + QC ───────────────────────────────────────────────
    logger.info("Step 1 | Preprocessing + QC  (%d runs)", len(runs_df))
    preproc_df = run_preprocess_qc(cfg, runs_df, logger)
    n_kept = int((~preproc_df["exclude"]).sum())
    n_excl = int(preproc_df["exclude"].sum())
    logger.info("         Kept %d / %d runs  (excluded %d for motion/coverage)", n_kept, len(preproc_df), n_excl)

    good_df = preproc_df[~preproc_df["exclude"]].copy()
    if good_df.empty:
        logger.error("All runs excluded after QC. Aborting.")
        return

    # ── Step 2: ROI time series ───────────────────────────────────────────────
    logger.info("Step 2 | Schaefer-200 ROI time series extraction")
    roi_df = run_roi_step(cfg, good_df, logger)

    # ── Step 3: ReHo ─────────────────────────────────────────────────────────
    logger.info("Step 3 | Regional Homogeneity (ReHo)")
    run_reho_step(cfg, good_df, logger)

    # ── Step 4: Static FC + dynamic FC ───────────────────────────────────────
    logger.info("Step 4 | Static FC + dynamic FC")
    run_static_dynamic_fc(cfg, roi_df, logger)

    # ── Step 5: Spatial ICA ───────────────────────────────────────────────────
    logger.info("Step 5 | Spatial ICA  (n_components=%d)", cfg["ica"]["n_components"])
    run_ica_step(cfg, good_df, logger)

    # ── Step 6: PCA on ROI time series ────────────────────────────────────────
    logger.info("Step 6 | PCA on ROI time series  (n_components=%d)", cfg["pca"]["n_components"])
    run_pca_step(cfg, roi_df, logger)

    # ── ISC and group_stats intentionally omitted ─────────────────────────────
    logger.info("ISC step skipped (requires naturalistic movie data, dataset=='algonauts')")
    logger.info("Group stats skipped (no control group — all subjects labelled 'patient')")

    logger.info("=" * 70)
    logger.info("Pipeline complete.  Outputs written to:  %s", output_root)
    logger.info("=" * 70)

    _print_output_summary(output_root)


def _print_output_summary(output_root: Path) -> None:
    """Print a concise tree of the most important output files."""
    print("\n── Output summary ─────────────────────────────────────────────────")
    key_dirs = [
        "preprocessed", "roi_timeseries", "reho", "static_fc", "dynamic_fc",
        "ica", "pca", "qc", "manifests", "logs",
    ]
    for d in key_dirs:
        p = output_root / d
        if p.exists():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {d:20s}  {n:4d} files")
    print("───────────────────────────────────────────────────────────────────\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ds007318 fMRIPrep pilot without BIDSLayout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "fmri-pipeline-data",
        help="Root directory of the fMRIPrep download (must contain a 'derivatives/' sub-folder). "
             "Default: ~/fmri-pipeline-data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write pipeline outputs. Default: <data-root>/pipeline_output",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.expanduser().resolve()
    output_root: Path = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else data_root / "pipeline_output"
    )

    if not data_root.exists():
        sys.exit(f"ERROR: data-root does not exist: {data_root}")

    deriv = data_root / "derivatives"
    if not deriv.exists():
        sys.exit(
            f"ERROR: expected a 'derivatives/' sub-folder under {data_root} but found none.\n"
            "       Make sure --data-root points to the fMRIPrep top-level directory."
        )

    print(f"Data root   : {data_root}")
    print(f"Output root : {output_root}")
    run_pilot(data_root, output_root)


if __name__ == "__main__":
    main()
