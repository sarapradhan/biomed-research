"""Schaefer atlas parcellation and ROI time series extraction.

Fetches the Schaefer 2018 local-global cortical parcellation at configurable
granularity (100, 200, or 400 ROIs mapped to 7 Yeo canonical networks) and
extracts mean BOLD time series per parcel using nilearn's NiftiLabelsMasker.

References
----------
Schaefer, A. et al. (2018). Local-global parcellation of the human cerebral
cortex from intrinsic functional connectivity MRI. Cerebral Cortex, 28(9),
3095-3114.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker


def get_schaefer_atlas(cfg: Dict):
    """Fetch Schaefer-200 atlas image + labels."""
    sch = datasets.fetch_atlas_schaefer_2018(
        n_rois=int(cfg["roi"]["schaefer_n_rois"]),
        yeo_networks=int(cfg["roi"]["schaefer_yeo_networks"]),
        resolution_mm=2,
    )
    return sch.maps, sch.labels


def extract_roi_timeseries(clean_bold_file: str, atlas_img, tr: float, cfg: Dict) -> np.ndarray:
    """Extract standardized Schaefer ROI time series."""
    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        standardize=cfg["roi"].get("standardize", "zscore_sample"),
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=tr,
    )
    ts = masker.fit_transform(clean_bold_file)
    return ts


def save_roi_timeseries(ts: np.ndarray, labels, out_dir: Path) -> Tuple[str, str]:
    """Save ROI time series as NPY and CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / "roi_timeseries.npy"
    csv_path = out_dir / "roi_timeseries.csv"
    np.save(npy_path, ts)
    cols = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in labels]
    if len(cols) != ts.shape[1]:
        # Schaefer label vectors may include background; align to extracted ROI columns.
        cols = cols[-ts.shape[1]:]
    pd.DataFrame(ts, columns=cols).to_csv(csv_path, index=False)
    return str(npy_path), str(csv_path)


def check_roi_variance(
    ts: np.ndarray,
    labels,
    zero_var_threshold: float = 1e-10,
    low_var_threshold: float = 1e-4,
) -> pd.DataFrame:
    """Audit ROI timeseries for zero- and low-variance parcels.

    Zero-variance ROIs produce constant timeseries after parcellation, which
    causes undefined (NaN) Pearson correlations in all FC edges involving that
    parcel.  This check should be run immediately after ROI extraction, before
    any connectivity or ReHo computation.

    Parameters
    ----------
    ts : np.ndarray, shape (T, R)
        ROI timeseries matrix (timepoints × ROIs).
    labels : sequence
        ROI label strings (or bytes) of length R.
    zero_var_threshold : float
        Variance below this value is considered zero (default 1e-10).
    low_var_threshold : float
        Variance below this value (but >= zero_var_threshold) triggers a
        low-variance warning (default 1e-4).

    Returns
    -------
    pd.DataFrame with columns:
        roi_idx, roi_label, variance, std, is_zero_var, is_low_var
    """
    variances = ts.var(axis=0)
    stds = ts.std(axis=0)
    str_labels = [
        lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl)
        for lbl in labels
    ]
    # Align label length to ROI count (Schaefer includes background label)
    if len(str_labels) != len(variances):
        str_labels = str_labels[-len(variances):]

    return pd.DataFrame(
        {
            "roi_idx": np.arange(len(variances)),
            "roi_label": str_labels,
            "variance": variances,
            "std": stds,
            "is_zero_var": variances < zero_var_threshold,
            "is_low_var": (variances >= zero_var_threshold) & (variances < low_var_threshold),
        }
    )


def save_roi_qc_report(report_df: pd.DataFrame, out_dir: Path) -> str:
    """Save per-run ROI QC report as CSV and return the file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "roi_qc_report.csv"
    report_df.to_csv(path, index=False)
    return str(path)

