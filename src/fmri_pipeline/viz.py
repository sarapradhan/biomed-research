from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


def save_thresholded_diff_matrix(beta_vec: np.ndarray, q_vec: np.ndarray, edge_idx, n_rois: int, out_file: str, title: str) -> None:
    """Render thresholded matrix of significant diagnosis effects."""
    mat = np.zeros((n_rois, n_rois), dtype=float)
    sig = q_vec < 0.05
    i, j = edge_idx
    mat[(i[sig], j[sig])] = beta_vec[sig]
    mat[(j[sig], i[sig])] = beta_vec[sig]

    vmax = np.percentile(np.abs(mat), 99) if np.any(mat) else 1.0
    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_box_by_group(df: pd.DataFrame, x: str, y: str, out_file: str, title: str) -> None:
    """Simple boxplot by categorical grouping."""
    if df.empty or x not in df.columns or y not in df.columns:
        return
    plt.figure(figsize=(7, 4))
    groups = [g[y].dropna().to_numpy() for _, g in df.groupby(x)]
    labels = [str(k) for k, _ in df.groupby(x)]
    if not groups:
        plt.close()
        return
    plt.boxplot(groups, labels=labels)
    plt.title(title)
    plt.ylabel(y)
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_voxel_map(map_file: str, out_file: str, title: str, threshold: float = 0.0) -> None:
    """Render stat map figure as PNG."""
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    display = plotting.plot_stat_map(
        map_file,
        threshold=threshold,
        display_mode="ortho",
        cut_coords=(0, -20, 40),
        title=title,
        colorbar=True,
    )
    display.savefig(out_file)
    display.close()


def save_voxel_from_vector(vec: np.ndarray, reference_mask_file: str, out_file: str) -> None:
    """Write masked vector back to NIfTI using reference mask."""
    mask_img = nib.load(reference_mask_file)
    mask = mask_img.get_fdata() > 0
    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = vec.astype(np.float32)
    nib.save(nib.Nifti1Image(vol, mask_img.affine, mask_img.header), out_file)
