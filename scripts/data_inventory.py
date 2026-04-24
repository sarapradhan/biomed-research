"""Walk a BIDS directory and produce a per-run data inventory CSV.

Records exactly which subjects, sessions, and runs feed each downstream
analysis. Output matches the schema in
``docs/baseline/data_inventory_template.csv``.

Usage
-----
    python scripts/data_inventory.py \
        --bids-root /path/to/bids_dataset \
        --dataset-name ds007318 \
        --output docs/baseline/data_inventory.ds007318.csv

Optionally include fMRIPrep-derived motion (mean FD and percent
censored) by passing ``--fmriprep-root``. If absent the QC columns are
left blank and can be filled in later.

Notes
-----
This script is intentionally dependency-light: it uses only ``pybids``
for BIDS traversal and ``pandas`` for CSV writing, both already in
requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


INVENTORY_COLUMNS = [
    "dataset",
    "subject_id",
    "session_id",
    "task",
    "run_id",
    "n_trs",
    "tr_seconds",
    "voxel_size_mm",
    "space",
    "desc",
    "mean_fd_mm",
    "pct_censored",
    "passes_qc",
    "exclusion_reason",
    "notes",
]


@dataclass
class InventoryRow:
    dataset: str
    subject_id: str
    session_id: Optional[str]
    task: Optional[str]
    run_id: Optional[str]
    n_trs: Optional[int]
    tr_seconds: Optional[float]
    voxel_size_mm: Optional[str]
    space: Optional[str]
    desc: Optional[str]
    mean_fd_mm: Optional[float] = None
    pct_censored: Optional[float] = None
    passes_qc: Optional[bool] = None
    exclusion_reason: Optional[str] = None
    notes: str = ""


def _safe_get(metadata: dict, key: str):
    try:
        return metadata[key]
    except (KeyError, TypeError):
        return None


def build_inventory(
    bids_root: Path,
    dataset_name: str,
    fmriprep_root: Optional[Path] = None,
) -> List[InventoryRow]:
    """Iterate bold runs and return one row per run."""
    try:
        from bids import BIDSLayout
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "pybids is required. Install with `pip install pybids`."
        ) from e

    layout = BIDSLayout(str(bids_root), validate=False)
    bolds = layout.get(suffix="bold", extension=["nii", "nii.gz"])

    rows: List[InventoryRow] = []
    for img in bolds:
        meta = img.get_metadata() or {}
        entities = img.get_entities()
        voxel = None
        try:
            import nibabel as nib
            nii = nib.load(img.path)
            zooms = nii.header.get_zooms()
            if len(zooms) >= 3:
                voxel = "x".join(f"{z:.2f}" for z in zooms[:3])
            n_trs = int(nii.shape[3]) if nii.ndim == 4 else None
        except Exception:  # noqa: BLE001
            n_trs = None

        rows.append(
            InventoryRow(
                dataset=dataset_name,
                subject_id=entities.get("subject"),
                session_id=entities.get("session"),
                task=entities.get("task"),
                run_id=entities.get("run"),
                n_trs=n_trs,
                tr_seconds=_safe_get(meta, "RepetitionTime"),
                voxel_size_mm=voxel,
                space=entities.get("space"),
                desc=entities.get("desc"),
            )
        )

    if fmriprep_root is not None:
        _augment_with_fmriprep_qc(rows, fmriprep_root)

    return rows


def _augment_with_fmriprep_qc(rows: List[InventoryRow], fmriprep_root: Path) -> None:
    """Fill mean_fd_mm and pct_censored from fMRIPrep confounds tsvs where possible.

    This is a best-effort scanner; it silently leaves values blank when
    the expected files are missing.
    """
    import numpy as np

    for row in rows:
        if row.subject_id is None or row.task is None:
            continue
        parts = [f"sub-{row.subject_id}"]
        if row.session_id:
            parts.append(f"ses-{row.session_id}")
        confounds_dir = fmriprep_root.joinpath(*parts, "func")
        if not confounds_dir.exists():
            continue
        base = f"sub-{row.subject_id}"
        if row.session_id:
            base += f"_ses-{row.session_id}"
        base += f"_task-{row.task}"
        if row.run_id:
            base += f"_run-{row.run_id}"
        candidate = confounds_dir / f"{base}_desc-confounds_timeseries.tsv"
        if not candidate.exists():
            continue
        try:
            df = pd.read_csv(candidate, sep="\t")
            if "framewise_displacement" in df.columns:
                fd = df["framewise_displacement"].astype(float)
                row.mean_fd_mm = float(fd.mean(skipna=True))
                row.pct_censored = float((fd > 0.5).sum() / len(fd)) if len(fd) else None
        except Exception:  # noqa: BLE001
            continue


def to_dataframe(rows: List[InventoryRow]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(r) for r in rows])
    # Enforce column order so the CSV schema stays stable.
    for col in INVENTORY_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[INVENTORY_COLUMNS]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bids-root", type=Path, required=True)
    ap.add_argument("--dataset-name", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--fmriprep-root",
        type=Path,
        default=None,
        help="Optional path to fMRIPrep derivatives for mean-FD enrichment.",
    )
    args = ap.parse_args(argv)

    if not args.bids_root.exists():
        print(f"ERROR: bids-root does not exist: {args.bids_root}", file=sys.stderr)
        return 2

    rows = build_inventory(
        bids_root=args.bids_root,
        dataset_name=args.dataset_name,
        fmriprep_root=args.fmriprep_root,
    )
    df = to_dataframe(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    manifest = {
        "dataset": args.dataset_name,
        "bids_root": str(args.bids_root),
        "fmriprep_root": str(args.fmriprep_root) if args.fmriprep_root else None,
        "n_runs": len(rows),
        "output": str(args.output),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
