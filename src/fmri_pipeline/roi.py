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
