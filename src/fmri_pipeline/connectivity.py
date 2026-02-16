from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.vq import kmeans2


def static_fc(roi_ts: np.ndarray) -> np.ndarray:
    """Compute ROI Pearson correlation matrix and Fisher-z transform."""
    corr = np.corrcoef(roi_ts.T)
    np.fill_diagonal(corr, 0.0)
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    np.fill_diagonal(z, 0.0)
    return z


def sliding_windows(roi_ts: np.ndarray, window_trs: int, step_trs: int) -> List[np.ndarray]:
    """Generate dynamic FC matrices for sliding windows."""
    t = roi_ts.shape[0]
    mats = []
    for start in range(0, t - window_trs + 1, step_trs):
        chunk = roi_ts[start : start + window_trs]
        mats.append(static_fc(chunk))
    return mats


def dynamic_fc_summary(roi_ts: np.ndarray, cfg: Dict) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute mean and variability (std) dFC matrices across windows."""
    w = int(cfg["dynamic_fc"]["window_trs"])
    s = int(cfg["dynamic_fc"]["step_trs"])
    mats = sliding_windows(roi_ts, w, s)
    if not mats:
        n = roi_ts.shape[1]
        zero = np.zeros((n, n), dtype=float)
        return zero, zero, []
    stack = np.stack(mats, axis=0)
    return np.mean(stack, axis=0), np.std(stack, axis=0), mats


def exploratory_window_clustering(mats: List[np.ndarray], cfg: Dict) -> Dict[str, np.ndarray]:
    """Optional exploratory clustering of window states."""
    if not mats:
        return {"labels": np.array([]), "centroids": np.array([])}
    k = int(cfg["dynamic_fc"].get("exploratory_clusters", 4))
    triu = np.triu_indices(mats[0].shape[0], k=1)
    x = np.stack([m[triu] for m in mats], axis=0)
    centroids, labels = kmeans2(x, k, minit="points")
    return {"labels": labels, "centroids": centroids}


def save_matrix(mat: np.ndarray, out_dir: Path, stem: str) -> str:
    """Save matrix as NPY and CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / f"{stem}.npy"
    csv = out_dir / f"{stem}.csv"
    np.save(npy, mat)
    np.savetxt(csv, mat, delimiter=",")
    return str(npy)


def plot_matrix(mat: np.ndarray, out_file: str, title: str) -> None:
    """Plot connectivity matrix."""
    plt.figure(figsize=(6, 5))
    vmax = np.percentile(np.abs(mat), 99) if np.any(mat) else 1.0
    plt.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()
