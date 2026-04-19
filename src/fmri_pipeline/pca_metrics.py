"""Principal component analysis of parcellated fMRI time series.

Computes the top-k explained variance ratios from PCA applied to
concatenated Schaefer ROI time series per subject, providing a concise
summary of effective dimensionality in the functional data.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run_subject_pca(roi_ts: np.ndarray, cfg: Dict) -> np.ndarray:
    """Return top-k explained variance ratios from PCA on ROI series."""
    n_req = int(cfg["pca"].get("n_components", 5))
    n_fit = min(n_req, min(roi_ts.shape[0], roi_ts.shape[1]))
    pca = PCA(n_components=n_fit, random_state=cfg["project"]["random_seed"])
    pca.fit(roi_ts)
    evr = pca.explained_variance_ratio_
    if evr.shape[0] < n_req:
        evr = np.pad(evr, (0, n_req - evr.shape[0]))
    return evr


def append_pca_row(rows: list, subject: str, dataset: str, diagnosis: str, evr: np.ndarray) -> None:
    """Append tidy PCA summary rows."""
    for i, v in enumerate(evr, start=1):
        rows.append(
            {
                "subject": subject,
                "dataset": dataset,
                "diagnosis": diagnosis,
                "component": i,
                "explained_variance_ratio": float(v),
            }
        )


def save_pca_table(rows: list, out_dir: Path) -> str:
    """Save PCA explained variance table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / "pca_explained_variance.csv"
    if 'subject' in df.columns:
        df = df.copy()
        df['subject'] = df['subject'].astype(str)
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return str(path)
