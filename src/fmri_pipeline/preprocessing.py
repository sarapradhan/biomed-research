"""BOLD preprocessing: confound regression, motion scrubbing, filtering, and smoothing.

Implements a denoising strategy following established connectivity-analysis
best practices: Friston-24 motion parameter expansion, white-matter and CSF
nuisance regression, optional global signal regression (GSR), framewise
displacement-based scrubbing, temporal bandpass filtering, and spatial
Gaussian smoothing. Also provides DVARS and tSNR quality metrics.

References
----------
Power, J. D. et al. (2012). Spurious but systematic correlations in functional
connectivity MRI networks arise from subject motion. NeuroImage, 59(3), 2142-2154.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import smooth_img
from nilearn.maskers import NiftiMasker


def build_friston24(confounds_df: pd.DataFrame) -> pd.DataFrame:
    """Create Friston-24 matrix from 6 rigid-body motion parameters."""
    motion_cols = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    missing = [c for c in motion_cols if c not in confounds_df.columns]
    if missing:
        raise ValueError(f"Missing motion columns for Friston-24: {missing}")

    motion = confounds_df[motion_cols].fillna(0.0)
    deriv = motion.diff().fillna(0.0)
    squares = motion.pow(2)
    deriv_squares = deriv.pow(2)

    out = pd.concat([motion, deriv, squares, deriv_squares], axis=1)
    out.columns = (
        [f"{c}" for c in motion_cols]
        + [f"d_{c}" for c in motion_cols]
        + [f"sq_{c}" for c in motion_cols]
        + [f"sq_d_{c}" for c in motion_cols]
    )
    return out


def select_wm_csf(confounds_df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Select WM/CSF columns from available confounds."""
    if not cfg["preprocessing"]["confounds"].get("include_wm_csf", True):
        return pd.DataFrame(index=confounds_df.index)

    wm_candidates = cfg["preprocessing"]["confounds"].get("wm_columns", ["white_matter", "wm"])
    csf_candidates = cfg["preprocessing"]["confounds"].get("csf_columns", ["csf"])

    cols = []
    for c in wm_candidates + csf_candidates:
        if c in confounds_df.columns:
            cols.append(c)
    return confounds_df[cols].fillna(0.0) if cols else pd.DataFrame(index=confounds_df.index)


def build_scrub_mask(confounds_df: pd.DataFrame, fd_threshold: float) -> Tuple[np.ndarray, pd.Series]:
    """Return censor mask (True kept) and framewise displacement series."""
    if "framewise_displacement" not in confounds_df.columns:
        fd = pd.Series(np.zeros(len(confounds_df)), name="framewise_displacement")
    else:
        fd = confounds_df["framewise_displacement"].fillna(0.0)

    keep = (fd <= fd_threshold).to_numpy()
    return keep, fd


def compute_motion_exclusion(confounds_df: pd.DataFrame, keep_mask: np.ndarray, cfg: Dict) -> Dict[str, float]:
    """Compute exclusion flags using threshold rules."""
    scr_cfg = cfg["preprocessing"]["scrubbing"]

    trans_cols = [c for c in ["trans_x", "trans_y", "trans_z"] if c in confounds_df.columns]
    rot_cols = [c for c in ["rot_x", "rot_y", "rot_z"] if c in confounds_df.columns]

    max_trans = float(confounds_df[trans_cols].abs().max().max()) if trans_cols else 0.0
    max_rot_rad = float(confounds_df[rot_cols].abs().max().max()) if rot_cols else 0.0
    max_rot_deg = max_rot_rad * 180.0 / np.pi

    censored_frac = float((~keep_mask).mean())
    exclude = (
        censored_frac > float(scr_cfg["exclude_percent_censored"])
        or max_trans > float(scr_cfg["exclude_max_motion_mm"])
        or max_rot_deg > float(scr_cfg["exclude_max_rotation_deg"])
    )
    return {
        "percent_scrubbed": censored_frac,
        "max_translation_mm": max_trans,
        "max_rotation_deg": max_rot_deg,
        "exclude": bool(exclude),
    }


def extract_global_signal(bold_file: str, mask_file: str) -> pd.DataFrame:
    """Compute whole-brain global signal time series from masked BOLD data.

    The global signal regressor is the mean BOLD intensity across all in-mask
    voxels at each timepoint. Including it in the confound model (global signal
    regression, GSR) removes a global noise floor that is common to all voxels
    and can inflate apparent connectivity. GSR is recommended when the primary
    interest is in relative (rather than absolute) connectivity patterns.
    """
    masker = NiftiMasker(mask_img=mask_file, standardize=False)
    ts = masker.fit_transform(bold_file)  # [time, vox]
    gs = ts.mean(axis=1)                  # [time] — mean over all voxels
    return pd.DataFrame({"global_signal": gs})


def load_confound_matrix(
    confounds_file: str,
    cfg: Dict,
    bold_file: Optional[str] = None,
    mask_file: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.Series, Dict[str, float]]:
    """Load confounds and produce final regression matrix + scrub mask + metrics.

    When bold_file and mask_file are supplied and cfg has
    preprocessing.confounds.include_global_signal: true, the global signal
    regressor is appended to the confound matrix (GSR).
    """
    confounds_df = pd.read_csv(confounds_file, sep="\t")
    fr24 = build_friston24(confounds_df)
    wmcsf = select_wm_csf(confounds_df, cfg)

    parts = [fr24, wmcsf]

    include_gsr = cfg["preprocessing"]["confounds"].get("include_global_signal", False)
    if include_gsr and bold_file and mask_file:
        gs_df = extract_global_signal(bold_file, mask_file)
        # Align length in case BOLD has been trimmed relative to confounds TSV
        gs_df = gs_df.iloc[: len(fr24)].reset_index(drop=True)
        parts.append(gs_df)

    confound_matrix = pd.concat(parts, axis=1)

    keep, fd = build_scrub_mask(confounds_df, float(cfg["preprocessing"]["scrubbing"]["fd_threshold_mm"]))
    motion_metrics = compute_motion_exclusion(confounds_df, keep, cfg)
    return confound_matrix, keep, fd, motion_metrics


def preprocess_bold(
    bold_file: str,
    mask_file: str,
    confound_matrix: pd.DataFrame,
    keep_mask: np.ndarray,
    tr: float,
    cfg: Dict,
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """Confound regression + scrubbing + filtering + output unsmoothed and 6mm-smoothed images."""
    filt_cfg = cfg["preprocessing"]["filtering"]

    sample_mask = np.where(keep_mask)[0]
    if sample_mask.size < 20:
        raise ValueError("Too few uncensored volumes after scrubbing")

    masker = NiftiMasker(mask_img=mask_file, standardize=False)
    cleaned_ts = masker.fit_transform(
        bold_file,
        confounds=confound_matrix.to_numpy(),
        sample_mask=sample_mask,
    )

    clean_masker = NiftiMasker(
        mask_img=mask_file,
        standardize=False,
        detrend=True,
        t_r=tr,
        low_pass=float(filt_cfg["high_hz"]),
        high_pass=float(filt_cfg["low_hz"]),
    )
    filtered_ts = clean_masker.fit_transform(masker.inverse_transform(cleaned_ts))
    unsmoothed_img = clean_masker.inverse_transform(filtered_ts)

    fwhm = float(cfg["preprocessing"]["smoothing_fwhm_mm"])
    smoothed_img = smooth_img(unsmoothed_img, fwhm=fwhm)
    return unsmoothed_img, smoothed_img


def compute_dvars(cleaned_4d: np.ndarray) -> float:
    """Compute mean DVARS proxy from cleaned 4D array [x,y,z,t]."""
    diff = np.diff(cleaned_4d, axis=3)
    dvars_t = np.sqrt(np.mean(diff**2, axis=(0, 1, 2)))
    return float(np.mean(dvars_t)) if dvars_t.size else 0.0


def compute_tsnr(cleaned_4d: np.ndarray, mask: np.ndarray) -> float:
    """Compute tSNR proxy = mean(signal)/std(signal) over masked voxels."""
    vox = cleaned_4d[mask > 0]
    if vox.size == 0:
        return 0.0
    means = np.mean(vox, axis=1)
    stds = np.std(vox, axis=1)
    tsnr = np.divide(means, stds, out=np.zeros_like(means), where=stds > 0)
    return float(np.nanmean(tsnr))


def save_preprocessed_images(unsmoothed_img: nib.Nifti1Image, smoothed_img: nib.Nifti1Image, out_dir: Path) -> Dict[str, str]:
    """Save preprocessed images to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    unsm_path = out_dir / "clean_unsmoothed_bold.nii.gz"
    sm_path = out_dir / "clean_smoothed6mm_bold.nii.gz"
    nib.save(unsmoothed_img, str(unsm_path))
    nib.save(smoothed_img, str(sm_path))
    return {"unsmoothed": str(unsm_path), "smoothed": str(sm_path)}
