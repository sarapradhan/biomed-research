from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def run_subject_pca(roi_ts: np.ndarray, cfg: Dict) -> np.ndarray:
    """Return top-k explained variance ratios from PCA on ROI series."""
    n = int(cfg["pca"].get("n_components", 5))
    pca = PCA(n_components=n, random_state=cfg["project"]["random_seed"])
    pca.fit(roi_ts)
    evr = pca.explained_variance_ratio_
    if evr.shape[0] < n:
        evr = np.pad(evr, (0, n - evr.shape[0]))
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
    df.to_csv(path, index=False)
    return str(path)
