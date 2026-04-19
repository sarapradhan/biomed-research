#!/usr/bin/env python3
"""
organize_schizconnect.py

Utility to set up and validate the SchizConnect dataset directory structure
expected by the fMRI pipeline. Supports two modes:

  --validate   Check the existing structure and report missing files.
  --link       Symlink an existing fMRIPrep derivatives tree into the
               canonical derivatives/schizconnect/fmriprep/ location.

Usage examples:
  # Validate a freshly downloaded + pre-processed dataset
  python scripts/organize_schizconnect.py --config config/pipeline.yaml --validate

  # Symlink the pilot fMRIPrep tree (SZ patients) into the canonical path
  python scripts/organize_schizconnect.py --config config/pipeline.yaml \
      --link /path/to/existing/fmriprep/derivatives

  # Generate a participants.tsv template from the derivative subjects
  python scripts/organize_schizconnect.py --config config/pipeline.yaml \
      --make-participants
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Files every subject-session must have inside ses-X/func/ (or sub-level)
_REQUIRED_PATTERNS = [
    "*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
    "*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz",
    "*_desc-confounds_timeseries.tsv",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _find_file(subject_dir: Path, pattern: str) -> list[Path]:
    """Search subject dir and its func sub-directories for a pattern."""
    matches = list(subject_dir.rglob(pattern))
    return matches


def validate(cfg: dict) -> bool:
    dataset_name = "schizconnect"
    bids_root = Path(cfg["paths"]["bids_roots"][dataset_name])
    deriv_root = Path(cfg["paths"]["derivatives_root"]) / dataset_name / "fmriprep"

    ok = True

    # ---- participants.tsv --------------------------------------------------
    pts_path = bids_root / "participants.tsv"
    if not pts_path.exists():
        log.error("Missing participants.tsv at %s", pts_path)
        ok = False
    else:
        import pandas as pd
        pts = pd.read_csv(pts_path, sep="\t")
        diag_col = cfg["stats"].get("diagnosis_column", "group")
        if diag_col not in pts.columns:
            log.error("participants.tsv missing '%s' column (needed for group stats)", diag_col)
            ok = False
        else:
            sz_label = cfg["stats"].get("patient_label", "patient")
            hc_label = cfg["stats"].get("control_label", "control")
            sz_n = (pts[diag_col] == sz_label).sum()
            hc_n = (pts[diag_col] == hc_label).sum()
            log.info("participants.tsv: %d SZ (%s), %d HC (%s)", sz_n, sz_label, hc_n, hc_label)
            if sz_n == 0:
                log.warning("No SZ subjects found in participants.tsv")
            if hc_n == 0:
                log.warning("No HC subjects found in participants.tsv — group stats will be skipped")

    # ---- derivatives -------------------------------------------------------
    if not deriv_root.exists():
        log.error("Derivatives directory missing: %s", deriv_root)
        return False

    subjects = sorted([d for d in deriv_root.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    if not subjects:
        log.error("No subject directories found under %s", deriv_root)
        return False

    log.info("Found %d subject directories in derivatives", len(subjects))

    missing_counts: dict[str, int] = {p: 0 for p in _REQUIRED_PATTERNS}
    for sub_dir in subjects:
        for pattern in _REQUIRED_PATTERNS:
            if not _find_file(sub_dir, pattern):
                missing_counts[pattern] += 1
                log.debug("MISSING  %s  in  %s", pattern, sub_dir.name)

    for pattern, n_missing in missing_counts.items():
        if n_missing:
            log.warning("Pattern '%s' missing for %d/%d subjects", pattern, n_missing, len(subjects))
            ok = False
        else:
            log.info("OK  '%s'  present for all %d subjects", pattern, len(subjects))

    if ok:
        log.info("Validation PASSED — structure looks good for the pipeline.")
    else:
        log.error("Validation FAILED — fix the issues above before running the pipeline.")
    return ok


# ---------------------------------------------------------------------------
# Symlink helper
# ---------------------------------------------------------------------------

def link_derivatives(cfg: dict, src_root: str) -> None:
    """Symlink each sub-XXX directory from src_root into the canonical derivatives path."""
    dataset_name = "schizconnect"
    src = Path(src_root).resolve()
    dest = Path(cfg["paths"]["derivatives_root"]) / dataset_name / "fmriprep"

    if not src.exists():
        log.error("Source directory does not exist: %s", src)
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in src.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    if not subject_dirs:
        log.warning("No sub-* directories found under %s", src)
        return

    linked = 0
    for sub_dir in subject_dirs:
        target = dest / sub_dir.name
        if target.exists() or target.is_symlink():
            log.info("Already exists, skipping: %s", target)
            continue
        target.symlink_to(sub_dir)
        log.info("Linked  %s  →  %s", target, sub_dir)
        linked += 1

    # Also link dataset_description.json if present
    for fname in ["dataset_description.json"]:
        src_file = src / fname
        if src_file.exists():
            dst_file = dest / fname
            if not dst_file.exists():
                dst_file.symlink_to(src_file)

    log.info("Done — %d subject directories linked into %s", linked, dest)


# ---------------------------------------------------------------------------
# participants.tsv generator
# ---------------------------------------------------------------------------

def make_participants(cfg: dict) -> None:
    """Generate a participants.tsv template from subject dirs in derivatives."""
    import pandas as pd

    dataset_name = "schizconnect"
    bids_root = Path(cfg["paths"]["bids_roots"][dataset_name])
    deriv_root = Path(cfg["paths"]["derivatives_root"]) / dataset_name / "fmriprep"
    out_path = bids_root / "participants.tsv"

    diag_col = cfg["stats"].get("diagnosis_column", "group")
    sz_label = cfg["stats"].get("patient_label", "patient")

    if not deriv_root.exists():
        log.error("Derivatives not found at %s — run --link first or ensure data is present", deriv_root)
        sys.exit(1)

    subject_dirs = sorted([d.name for d in deriv_root.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])

    rows = []
    for sub in subject_dirs:
        rows.append({
            "participant_id": sub,
            "age": "",
            "sex": "",
            diag_col: sz_label,   # default to SZ — EDIT for HC subjects
        })

    df = pd.DataFrame(rows, columns=["participant_id", "age", "sex", diag_col])

    if out_path.exists():
        log.warning("participants.tsv already exists at %s — overwrite? (y/n)", out_path)
        if input().strip().lower() != "y":
            log.info("Aborted.")
            return

    df.to_csv(out_path, sep="\t", index=False)
    log.info("Wrote participants.tsv with %d subjects to %s", len(df), out_path)
    log.info("ACTION REQUIRED: Open the file and set '%s' to '%s' (HC label) for healthy controls.",
             diag_col, cfg["stats"].get("control_label", "control"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SchizConnect dataset organizer and validator")
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate", action="store_true",
                       help="Validate directory structure and participants.tsv")
    group.add_argument("--link", metavar="SRC_DERIV_ROOT",
                       help="Symlink existing fMRIPrep derivatives tree into canonical path")
    group.add_argument("--make-participants", action="store_true",
                       help="Generate a participants.tsv template from derivative subjects")

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.validate:
        ok = validate(cfg)
        sys.exit(0 if ok else 1)
    elif args.link:
        link_derivatives(cfg, args.link)
    elif args.make_participants:
        make_participants(cfg)


if __name__ == "__main__":
    main()
