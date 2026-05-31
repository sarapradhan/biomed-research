#!/usr/bin/env python3
"""prep_reproducibility_inputs.py — Stage pipeline outputs for reproducibility analyses.

Reads the existing ds007318 pipeline outputs and writes them
into the flat directory structure expected by the reproducibility modules
(``src/fmri_pipeline/reproducibility/``).

Mapping performed
-----------------
``static_fc/sub-XX/sub-XX_ses-N_task-removal/static_fc_fisherz.npy``
    → ``data/repro_inputs/connectivity/sub-XX_run-N_fc.npy``

``reho/sub-XX/sub-XX_ses-N_task-removal/reho_map.nii.gz``  (3-D volume)
    → ``data/repro_inputs/reho/sub-XX_run-N_reho.npy``  (1-D ROI vector)
    Uses the Schaefer-200 atlas (fetched via nilearn if not cached) to
    extract mean ReHo per parcel.

``roi_timeseries/sub-XX/sub-XX_ses-N_task-removal/roi_timeseries.npy``
    → ``data/repro_inputs/roi_timeseries/sub-XX_run-N_roi.npy``

ICA seed sweep & leave-one-run-out cross-validation (Step 1.3):
    Fits temporal FastICA on the group-concatenated ROI timeseries with
    five random seeds (seed sweep) and with each subject omitted
    (LORO-CV).  Components are saved as 2-D arrays (K × n_ROIs).

    METHODOLOGICAL NOTE: The main-pipeline spatial ICA (run on full BOLD
    volumes) cannot be efficiently re-run from a prep script.  This step
    therefore computes temporal ICA on the Schaefer-200 parcellated time
    series.  The component matrices are used *solely* to test stability
    of the decomposition across random initialisations and data subsets;
    they are not used as brain maps.  This is documented in the Methods
    section of the manuscript.

Usage
-----
    # From the fMRI-Enhance project root:
    python scripts/prep_reproducibility_inputs.py

    # Override the data root (defaults to /path/to/fmri-pipeline/derivatives/metrics):
    python scripts/prep_reproducibility_inputs.py \
        --data-root "/path/to/derivatives/metrics"

Outputs
-------
    data/repro_inputs/
        connectivity/          sub-*_run-*_fc.npy
        reho/                  sub-*_run-*_reho.npy
        roi_timeseries/        sub-*_run-*_roi.npy
        ica/
            seeds/             seed-{1..5}_ica_components.npy
            lorocv/            lorocv-sub{XX}_ica_components.npy
        atlas/                 roi_labels.csv

    data/repro_inputs/prep_manifest.json   (summary of what was staged)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = (
    Path("/path/to/fmri-pipeline") / "derivatives" / "metrics"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "repro_inputs"

# ds007318 subjects and their session→run mapping.
# Session labels used in file names are mapped to sequential run IDs
# (run-1, run-2, …) so that the reproducibility modules, which compare
# run pairs within and across subjects, see consistent identifiers.
SUBJECT_SESSION_MAP: Dict[str, List[str]] = {
    "sub-01": ["ses-1", "ses-2"],
    "sub-02": ["ses-1"],
    "sub-03": ["ses-1", "ses-2"],
}

N_ICA_COMPONENTS = 20
ICA_SEEDS = [1, 2, 3, 4, 5]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_id(session: str) -> str:
    """Convert 'ses-1' → 'run-1' (strip 'ses-' prefix, add 'run-')."""
    num = session.replace("ses-", "")
    return f"run-{num}"


def _flat_stem(subject: str, session: str) -> str:
    """Return e.g. 'sub-01_run-1'."""
    return f"{subject}_{_run_id(session)}"


# ── Stage 1: Static FC matrices ───────────────────────────────────────────────

def stage_fc(data_root: Path, out_dir: Path) -> List[str]:
    """Copy per-run Fisher-z FC matrices to flat connectivity directory."""
    _ensure(out_dir)
    staged: List[str] = []
    for sub, sessions in SUBJECT_SESSION_MAP.items():
        for ses in sessions:
            src = (
                data_root / "static_fc" / sub
                / f"{sub}_{ses}_task-removal" / "static_fc_fisherz.npy"
            )
            if not src.exists():
                log.warning("FC not found: %s", src)
                continue
            dst = out_dir / f"{_flat_stem(sub, ses)}_fc.npy"
            shutil.copy2(src, dst)
            mat = np.load(dst)
            log.info("FC   %s → %s  shape=%s", src.name, dst.name, mat.shape)
            staged.append(str(dst))
    return staged


# ── Stage 2: ReHo ROI vectors ─────────────────────────────────────────────────

def _get_schaefer_atlas(n_rois: int = 200, resolution: int = 2):
    """Fetch Schaefer parcellation via nilearn (cached after first download)."""
    try:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=resolution)
        return atlas["maps"]
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch Schaefer-{n_rois} atlas: {exc}"
        ) from exc


def _is_empty_nifti(path: Path) -> bool:
    """Return True if the NIfTI file is a stub (all-zero or size < 10 KB)."""
    import nibabel as nib
    if path.stat().st_size < 10_000:
        return True  # too small for a real brain volume
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(float)
    return bool(np.count_nonzero(data) == 0)


def _derive_brain_mask(bold_img) -> np.ndarray:
    """Return a boolean brain mask from temporal standard deviation of BOLD."""
    data = bold_img.get_fdata(dtype=np.float32)  # (x, y, z, t)
    std_map = data.std(axis=-1)
    threshold = np.percentile(std_map[std_map > 0], 5)
    return std_map > threshold


def _compute_reho_from_bold(
    bold_path: Path,
    n_jobs: int = 4,
) -> "nib.Nifti1Image":
    """Recompute ReHo (Kendall's W) from preprocessed BOLD.

    Derives a brain mask from temporal std of the BOLD signal and runs
    the pipeline's voxelwise ReHo computation.
    """
    import nibabel as nib
    import sys
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from fmri_pipeline.reho import compute_reho_map

    # Write a temporary mask NIfTI that compute_reho_map can load
    import tempfile
    bold_img = nib.load(str(bold_path))
    brain_mask = _derive_brain_mask(bold_img)
    mask_img = nib.Nifti1Image(brain_mask.astype(np.uint8), bold_img.affine, bold_img.header)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        mask_path = tmp.name
    nib.save(mask_img, mask_path)

    cfg = {"project": {"debug_mode": False}, "reho": {"chunk_size_voxels": 5000}}
    try:
        reho_img = compute_reho_map(str(bold_path), mask_path, cfg, n_jobs=n_jobs)
    finally:
        Path(mask_path).unlink(missing_ok=True)

    return reho_img


def _extract_roi_means(
    volume_path: Path,
    atlas_img,
    n_rois: int = 200,
) -> np.ndarray:
    """Return mean value within each ROI label of an atlas."""
    import nibabel as nib
    from nilearn.image import resample_to_img

    vol_img = nib.load(str(volume_path))
    atlas_r = resample_to_img(
        atlas_img, vol_img, interpolation="nearest", copy_header=True
    )
    atlas_data = atlas_r.get_fdata(dtype=np.float32).astype(int)
    vol_data = vol_img.get_fdata(dtype=np.float32)

    roi_means = np.zeros(n_rois, dtype=np.float64)
    for roi_idx in range(1, n_rois + 1):
        mask = atlas_data == roi_idx
        if mask.any():
            roi_means[roi_idx - 1] = float(vol_data[mask].mean())
    return roi_means


def stage_reho(
    data_root: Path,
    out_dir: Path,
    atlas_img,
    n_rois: int = 200,
    recompute_if_empty: bool = True,
) -> List[str]:
    """Extract per-ROI mean ReHo and save as 1-D .npy vectors.

    If the stored reho_map.nii.gz is an empty stub (all zeros, < 10 KB),
    and *recompute_if_empty* is True, recomputes ReHo from the
    corresponding preprocessed unsmoothed BOLD volume.
    """
    _ensure(out_dir)
    staged: List[str] = []
    for sub, sessions in SUBJECT_SESSION_MAP.items():
        for ses in sessions:
            src = (
                data_root / "reho" / sub
                / f"{sub}_{ses}_task-removal" / "reho_map.nii.gz"
            )
            dst = out_dir / f"{_flat_stem(sub, ses)}_reho.npy"

            reho_nii_to_use: Optional[Path] = None

            if not src.exists():
                log.warning("ReHo NIfTI not found: %s", src)
            elif _is_empty_nifti(src):
                log.warning("ReHo NIfTI is empty stub: %s", src)
                if recompute_if_empty:
                    bold_path = (
                        data_root / "preprocessed" / sub
                        / f"{sub}_{ses}_task-removal"
                        / "clean_unsmoothed_bold.nii.gz"
                    )
                    if bold_path.exists():
                        log.info("  Recomputing ReHo from BOLD: %s …", bold_path.name)
                        try:
                            import nibabel as nib
                            import tempfile
                            reho_img = _compute_reho_from_bold(bold_path)
                            with tempfile.NamedTemporaryFile(
                                suffix=".nii.gz", delete=False
                            ) as tmp:
                                tmp_nii = Path(tmp.name)
                            nib.save(reho_img, str(tmp_nii))
                            reho_nii_to_use = tmp_nii
                            log.info("  ReHo recomputed OK")
                        except Exception as exc:
                            log.error("  ReHo recompute failed: %s", exc)
                    else:
                        log.warning("  BOLD not found for recompute: %s", bold_path)
            else:
                reho_nii_to_use = src

            if reho_nii_to_use is None:
                log.warning("  Skipping ReHo for %s %s — no usable source", sub, ses)
                continue

            try:
                roi_vec = _extract_roi_means(reho_nii_to_use, atlas_img, n_rois=n_rois)
                np.save(dst, roi_vec)
                log.info(
                    "ReHo %s %s → %s  mean=%.4f",
                    sub, ses, dst.name, roi_vec.mean()
                )
                staged.append(str(dst))
            except Exception as exc:
                log.error("  ROI extraction failed for %s %s: %s", sub, ses, exc)
            finally:
                # Remove temp file if we created one
                if reho_nii_to_use != src and reho_nii_to_use is not None:
                    reho_nii_to_use.unlink(missing_ok=True)

    return staged


# ── Stage 3: ROI timeseries ───────────────────────────────────────────────────

def stage_roi_timeseries(data_root: Path, out_dir: Path) -> List[str]:
    """Copy per-run ROI timeseries matrices to flat roi_timeseries directory."""
    _ensure(out_dir)
    staged: List[str] = []
    for sub, sessions in SUBJECT_SESSION_MAP.items():
        for ses in sessions:
            src = (
                data_root / "roi_timeseries" / sub
                / f"{sub}_{ses}_task-removal" / "roi_timeseries.npy"
            )
            if not src.exists():
                log.warning("ROI ts not found: %s", src)
                continue
            dst = out_dir / f"{_flat_stem(sub, ses)}_roi.npy"
            shutil.copy2(src, dst)
            ts = np.load(dst)
            log.info(
                "ROI ts  %s → %s  shape=%s", src.name, dst.name, ts.shape
            )
            staged.append(str(dst))
    return staged


# ── Stage 4: ROI labels CSV ───────────────────────────────────────────────────

def stage_roi_labels(out_dir: Path, n_rois: int = 200) -> Path:
    """Write a minimal ROI labels CSV for the network_anchor module.

    Writes exactly *n_rois* rows (indices 1..n_rois).  The nilearn
    Schaefer atlas labels may include a 'Background' entry at index 0;
    this function strips it so the CSV has one row per parcel.
    """
    _ensure(out_dir)
    try:
        from nilearn.datasets import fetch_atlas_schaefer_2018
        atlas = fetch_atlas_schaefer_2018(n_rois=n_rois, resolution_mm=2)
        labels = list(atlas["labels"])
        # Decode bytes if needed
        if labels and isinstance(labels[0], bytes):
            labels = [lb.decode() for lb in labels]
        # Strip background label if present
        labels = [lb for lb in labels if lb.lower() not in ("background", "")]
        # Trim or pad to exactly n_rois
        if len(labels) > n_rois:
            labels = labels[:n_rois]
        elif len(labels) < n_rois:
            labels += [f"Parcel_{i + 1}" for i in range(len(labels), n_rois)]
    except Exception:
        # Fallback: generic labels
        labels = [f"Parcel_{i + 1}" for i in range(n_rois)]

    csv_path = out_dir / "roi_labels.csv"
    with csv_path.open("w") as f:
        f.write("roi_index,label\n")
        for i, label in enumerate(labels, start=1):
            f.write(f"{i},{label}\n")
    log.info("ROI labels → %s  (%d parcels)", csv_path, len(labels))
    return csv_path


# ── Stage 5: ICA seed sweep and LORO-CV ──────────────────────────────────────

def _load_group_roi_ts(
    data_root: Path,
    exclude_subject: Optional[str] = None,
) -> np.ndarray:
    """Concatenate ROI timeseries across all (or all-but-one) subjects/runs."""
    parts: List[np.ndarray] = []
    for sub, sessions in SUBJECT_SESSION_MAP.items():
        if sub == exclude_subject:
            continue
        for ses in sessions:
            path = (
                data_root / "roi_timeseries" / sub
                / f"{sub}_{ses}_task-removal" / "roi_timeseries.npy"
            )
            if path.exists():
                parts.append(np.load(path))
    if not parts:
        raise ValueError("No ROI timeseries found for ICA.")
    return np.concatenate(parts, axis=0)  # (total_TRs, n_ROIs)


def _run_ica(
    ts: np.ndarray,
    n_components: int,
    seed: int,
) -> np.ndarray:
    """Fit FastICA and return components matrix (K × n_features)."""
    from sklearn.decomposition import FastICA

    # Standardize each ROI timeseries before ICA
    ts_z = (ts - ts.mean(axis=0)) / (ts.std(axis=0, ddof=1) + 1e-8)

    ica = FastICA(
        n_components=n_components,
        random_state=seed,
        max_iter=500,
        tol=1e-4,
        whiten="unit-variance",
    )
    ica.fit(ts_z)
    return ica.components_  # (K, n_ROIs)


def stage_ica(
    data_root: Path,
    out_dir: Path,
    n_components: int = N_ICA_COMPONENTS,
    seeds: List[int] = None,
) -> List[str]:
    """Run temporal ICA with multiple seeds and LORO-CV; save component matrices.

    Component shape: (K, n_ROIs)

    NOTE: This is temporal ICA on parcellated time series, not the spatial
    ICA on full BOLD that is used in the main pipeline analysis.  It serves
    to test the stability of the decomposition across random initialisations
    and data subsets.  See manuscript Methods for details.
    """
    if seeds is None:
        seeds = ICA_SEEDS

    seeds_dir = _ensure(out_dir / "seeds")
    lorocv_dir = _ensure(out_dir / "lorocv")
    staged: List[str] = []

    # ── 5-seed sweep on full group data ──────────────────────────────────────
    log.info("Loading group-concatenated ROI timeseries for ICA seed sweep…")
    try:
        group_ts = _load_group_roi_ts(data_root)
        log.info("Group TS shape: %s", group_ts.shape)
    except Exception as exc:
        log.error("Could not load group ROI timeseries: %s", exc)
        return staged

    for seed in seeds:
        log.info("  ICA seed=%d …", seed)
        try:
            comps = _run_ica(group_ts, n_components=n_components, seed=seed)
            dst = seeds_dir / f"seed-{seed}_ica_components.npy"
            np.save(dst, comps)
            log.info("    → %s  shape=%s", dst.name, comps.shape)
            staged.append(str(dst))
        except Exception as exc:
            log.error("    seed=%d failed: %s", seed, exc)

    # ── LORO-CV: omit each subject in turn ───────────────────────────────────
    for sub in SUBJECT_SESSION_MAP:
        log.info("ICA LORO-CV: leaving out %s …", sub)
        try:
            lorocv_ts = _load_group_roi_ts(data_root, exclude_subject=sub)
            log.info("  Reduced TS shape: %s", lorocv_ts.shape)
            comps = _run_ica(lorocv_ts, n_components=n_components, seed=42)
            tag = sub.replace("sub-", "")
            dst = lorocv_dir / f"lorocv-{tag}_ica_components.npy"
            np.save(dst, comps)
            log.info("  → %s  shape=%s", dst.name, comps.shape)
            staged.append(str(dst))
        except Exception as exc:
            log.error("  LORO-CV %s failed: %s", sub, exc)

    return staged


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage pipeline outputs into reproducibility input layout."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Root of the pipeline metrics outputs "
            f"(default: {DEFAULT_DATA_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Where to write staged inputs (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--skip-ica",
        action="store_true",
        help="Skip ICA seed sweep and LORO-CV (saves ~2–5 min)",
    )
    parser.add_argument(
        "--skip-reho-recompute",
        action="store_true",
        help="Do not recompute ReHo from BOLD when stored NIfTI files are empty stubs",
    )
    parser.add_argument(
        "--n-rois",
        type=int,
        default=200,
        help="Schaefer ROI count for ReHo extraction (default: 200)",
    )
    args = parser.parse_args(argv)

    data_root: Path = args.data_root.expanduser().resolve()
    output_root: Path = args.output_root.expanduser().resolve()

    log.info("Data root   : %s", data_root)
    log.info("Output root : %s", output_root)

    if not data_root.exists():
        log.error("Data root not found: %s", data_root)
        sys.exit(1)

    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "stages": {},
    }

    # Stage 1 — Static FC
    log.info("=== Stage 1: Static FC matrices ===")
    fc_out = output_root / "connectivity"
    fc_staged = stage_fc(data_root, fc_out)
    manifest["stages"]["fc"] = {"n_files": len(fc_staged), "dir": str(fc_out)}
    log.info("  Staged %d FC matrices", len(fc_staged))

    # Stage 2 — ReHo ROI vectors (requires nilearn atlas)
    log.info("=== Stage 2: ReHo ROI vectors ===")
    try:
        log.info("  Fetching Schaefer-%d atlas…", args.n_rois)
        atlas_img = _get_schaefer_atlas(n_rois=args.n_rois, resolution=2)
        reho_out = output_root / "reho"
        reho_staged = stage_reho(
            data_root, reho_out, atlas_img, n_rois=args.n_rois,
            recompute_if_empty=not args.skip_reho_recompute,
        )
        manifest["stages"]["reho"] = {
            "n_files": len(reho_staged),
            "dir": str(reho_out),
            "atlas": f"Schaefer-{args.n_rois}",
        }
        log.info("  Staged %d ReHo vectors", len(reho_staged))
    except Exception as exc:
        log.error("  ReHo staging failed: %s", exc)
        manifest["stages"]["reho"] = {"error": str(exc)}

    # Stage 3 — ROI timeseries
    log.info("=== Stage 3: ROI timeseries ===")
    roi_ts_out = output_root / "roi_timeseries"
    ts_staged = stage_roi_timeseries(data_root, roi_ts_out)
    manifest["stages"]["roi_timeseries"] = {
        "n_files": len(ts_staged),
        "dir": str(roi_ts_out),
    }
    log.info("  Staged %d ROI timeseries", len(ts_staged))

    # Stage 4 — ROI labels
    log.info("=== Stage 4: ROI labels ===")
    try:
        atlas_dir = output_root / "atlas"
        roi_labels_path = stage_roi_labels(atlas_dir, n_rois=args.n_rois)
        manifest["stages"]["roi_labels"] = {"path": str(roi_labels_path)}
    except Exception as exc:
        log.error("  ROI labels staging failed: %s", exc)
        manifest["stages"]["roi_labels"] = {"error": str(exc)}

    # Stage 5 — ICA stability
    if not args.skip_ica:
        log.info("=== Stage 5: ICA seed sweep + LORO-CV ===")
        ica_out = output_root / "ica"
        ica_staged = stage_ica(data_root, ica_out)
        manifest["stages"]["ica"] = {
            "n_files": len(ica_staged),
            "dir": str(ica_out),
            "method": "temporal_ica_on_roi_timeseries",
            "n_components": N_ICA_COMPONENTS,
            "seeds": ICA_SEEDS,
        }
        log.info("  Staged %d ICA component files", len(ica_staged))
    else:
        log.info("=== Stage 5: ICA — SKIPPED (--skip-ica) ===")
        manifest["stages"]["ica"] = {"skipped": True}

    # Write manifest
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = output_root / "prep_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest → %s", manifest_path)

    # Print summary
    log.info("=== Summary ===")
    total = sum(
        v.get("n_files", 0)
        for v in manifest["stages"].values()
        if isinstance(v, dict)
    )
    log.info("Total files staged: %d", total)
    log.info("Output root: %s", output_root)
    log.info("")
    log.info("Next step:")
    log.info(
        "  python scripts/run_reproducibility.py "
        "--config config/reproducibility_real.yaml"
    )


if __name__ == "__main__":
    main()
