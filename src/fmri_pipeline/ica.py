from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import FastICA


def run_subject_spatial_ica(clean_bold_file: str, mask_file: str, cfg: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Run subject-level spatial ICA; returns (maps, timecourses)."""
    n_comp = int(cfg["ica"]["n_components"])
    max_iter = int(cfg["ica"].get("max_iter", 500))

    masker = NiftiMasker(mask_img=mask_file, standardize=True)
    x = masker.fit_transform(clean_bold_file)  # [time, vox]

    ica = FastICA(n_components=n_comp, random_state=cfg["project"]["random_seed"], max_iter=max_iter)
    timecourses = ica.fit_transform(x)
    maps = ica.mixing_.T  # [components, vox]
    return maps, timecourses


def save_subject_ica(maps: np.ndarray, timecourses: np.ndarray, out_dir: Path) -> Dict[str, str]:
    """Save subject ICA outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_path = out_dir / "ica_maps.npy"
    tc_path = out_dir / "ica_timecourses.npy"
    np.save(maps_path, maps)
    np.save(tc_path, timecourses)
    return {"maps": str(maps_path), "timecourses": str(tc_path)}


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def match_components_across_subjects(subject_maps: Dict[str, np.ndarray], cfg: Dict) -> Tuple[pd.DataFrame, np.ndarray]:
    """Match components via spatial correlation + hierarchical clustering."""
    rows = []
    comp_vectors = []
    for sub, maps in subject_maps.items():
        z = _zscore_rows(maps)
        for c in range(z.shape[0]):
            comp_vectors.append(z[c])
            rows.append({"subject": sub, "component": c})

    comp_mat = np.stack(comp_vectors, axis=0)
    corr = np.corrcoef(comp_mat)
    np.fill_diagonal(corr, 1.0)

    dist = 1.0 - np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    z_link = linkage(condensed, method="average")
    n_clusters = int(cfg["ica"]["n_components"])
    labels = fcluster(z_link, n_clusters, criterion="maxclust")

    match_df = pd.DataFrame(rows)
    match_df["cluster"] = labels

    centroids = []
    for cl in sorted(np.unique(labels)):
        centroids.append(comp_mat[labels == cl].mean(axis=0))
    return match_df, np.stack(centroids, axis=0)


def compute_subject_network_loadings(subject_maps: Dict[str, np.ndarray], centroids: np.ndarray) -> pd.DataFrame:
    """Compute per-subject loading per canonical cluster using max abs corr."""
    out = []
    cent = _zscore_rows(centroids)
    for sub, maps in subject_maps.items():
        mz = _zscore_rows(maps)
        corr = np.corrcoef(mz, cent)[: mz.shape[0], mz.shape[0] :]
        for k in range(cent.shape[0]):
            out.append(
                {
                    "subject": sub,
                    "network_cluster": int(k + 1),
                    "loading": float(np.max(np.abs(corr[:, k]))),
                }
            )
    return pd.DataFrame(out)
