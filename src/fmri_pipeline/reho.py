from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata


def _neighbors(coord: Tuple[int, int, int], shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    x, y, z = coord
    neigh = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                    neigh.append((nx, ny, nz))
    return neigh


def _kendalls_w(nei_ts: np.ndarray) -> float:
    """Kendall's W for matrix [voxels, timepoints]."""
    m, n = nei_ts.shape
    if m < 3 or n < 10:
        return 0.0
    ranks = rankdata(nei_ts, axis=0)
    r_i = np.sum(ranks, axis=0)
    s = np.sum((r_i - np.mean(r_i)) ** 2)
    denom = (m**2) * (n**3 - n)
    if denom <= 0:
        return 0.0
    return float(12.0 * s / denom)


def _compute_chunk(coords, mask, data):
    out = []
    shape = mask.shape
    for c in coords:
        neigh = _neighbors(c, shape)
        neigh = [n for n in neigh if mask[n]]
        if len(neigh) < 7:
            out.append((c, 0.0))
            continue
        ts = np.stack([data[n] for n in neigh], axis=0)
        out.append((c, _kendalls_w(ts)))
    return out


def compute_reho_map(clean_bold_file: str, gm_mask_file: str, cfg: Dict, n_jobs: int = 1):
    """Compute voxelwise ReHo (Kendall's W) in gray-matter mask."""
    img = nib.load(clean_bold_file)
    data = img.get_fdata(dtype=np.float32)
    gm = nib.load(gm_mask_file).get_fdata() > 0

    coords = list(map(tuple, np.argwhere(gm)))
    if cfg["project"].get("debug_mode") and cfg["project"].get("debug_subject_limit"):
        coords = coords[: int(cfg["project"].get("debug_subject_limit")) * 1000]

    chunk_size = int(cfg["reho"].get("chunk_size_voxels", 10000))
    chunks = [coords[i : i + chunk_size] for i in range(0, len(coords), chunk_size)]

    results = Parallel(n_jobs=n_jobs)(delayed(_compute_chunk)(chunk, gm, data) for chunk in chunks)
    reho = np.zeros(gm.shape, dtype=np.float32)
    for chunk in results:
        for c, w in chunk:
            reho[c] = w

    if cfg["reho"].get("normalize_zscore", True):
        vals = reho[gm]
        mu = vals.mean() if vals.size else 0.0
        sd = vals.std() if vals.size else 1.0
        sd = sd if sd > 0 else 1.0
        reho[gm] = (vals - mu) / sd

    return nib.Nifti1Image(reho, affine=img.affine, header=img.header)


def save_reho_map(reho_img: nib.Nifti1Image, out_dir: Path) -> str:
    """Save ReHo map as NIfTI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reho_map.nii.gz"
    nib.save(reho_img, str(path))
    return str(path)
