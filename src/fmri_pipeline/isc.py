from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nilearn.maskers import NiftiMasker
from statsmodels.stats.multitest import multipletests


def _zscore_t(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def _corr_time(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    az = _zscore_t(a)
    bz = _zscore_t(b)
    return np.mean(az * bz, axis=0)


def _load_and_align(clean_movie_files: List[str], mask_file: str) -> List[np.ndarray]:
    """Load and mask each movie file, then truncate all to the shortest run."""
    masker = NiftiMasker(mask_img=mask_file, standardize=False)
    ts_list = [masker.fit_transform(f) for f in clean_movie_files]  # each [time, vox]
    min_t = min(t.shape[0] for t in ts_list)
    return [t[:min_t] for t in ts_list]


def _loo_isc(ts_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute leave-one-out ISC from a pre-loaded list of [time, vox] arrays."""
    n = len(ts_list)
    loo = []
    for i in range(n):
        others = np.mean([ts_list[j] for j in range(n) if j != i], axis=0)
        loo.append(_corr_time(ts_list[i], others))
    loo_arr = np.stack(loo, axis=0)   # [subjects, vox]
    mean_map = loo_arr.mean(axis=0)
    return loo_arr, mean_map


def compute_leave_one_out_isc(
    clean_movie_files: List[str],
    mask_file: str,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Compute per-subject leave-one-out ISC and group mean ISC in mask space.

    Returns (loo_arr, mean_map, ts_list). The ts_list can be forwarded to
    permutation_pvalues to avoid re-reading files from disk.
    """
    ts_list = _load_and_align(clean_movie_files, mask_file)
    loo_arr, mean_map = _loo_isc(ts_list)
    return loo_arr, mean_map, ts_list


def permutation_pvalues(
    clean_movie_files: List[str],
    mask_file: str,
    observed_mean: np.ndarray,
    cfg: Dict,
    ts_list: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Circular time-shift permutation null for mean ISC.

    If ts_list is supplied (pre-loaded from compute_leave_one_out_isc), the
    files are not re-read from disk, avoiding redundant I/O.
    """
    rng = np.random.default_rng(int(cfg["isc"].get("random_state", 42)))
    n_perm = int(cfg["isc"].get("permutations", 1000))

    if ts_list is None:
        ts_list = _load_and_align(clean_movie_files, mask_file)

    min_t = ts_list[0].shape[0]
    null = np.zeros((n_perm, observed_mean.shape[0]), dtype=np.float32)
    n = len(ts_list)

    for p in range(n_perm):
        shifted = []
        for t in ts_list:
            shift = int(rng.integers(0, min_t))
            shifted.append(np.roll(t, shift=shift, axis=0))

        loo = []
        for i in range(n):
            others = np.mean([shifted[j] for j in range(n) if j != i], axis=0)
            loo.append(_corr_time(shifted[i], others))
        null[p] = np.mean(np.stack(loo, axis=0), axis=0)

    pvals = np.mean(np.abs(null) >= np.abs(observed_mean[np.newaxis, :]), axis=0)
    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return pvals, qvals


def save_isc_maps(mean_vec: np.ndarray, p_vec: np.ndarray, q_vec: np.ndarray, mask_file: str, out_dir: Path) -> Dict[str, str]:
    """Save ISC mean, p, and q maps as NIfTI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    masker = NiftiMasker(mask_img=mask_file)
    masker.fit()

    mean_img = masker.inverse_transform(mean_vec)
    p_img = masker.inverse_transform(p_vec)
    q_img = masker.inverse_transform(q_vec)
    sig_img = masker.inverse_transform((q_vec < 0.05).astype(float) * mean_vec)

    paths = {
        "isc_mean": out_dir / "isc_mean.nii.gz",
        "isc_p": out_dir / "isc_pvals.nii.gz",
        "isc_q": out_dir / "isc_qvals.nii.gz",
        "isc_sig": out_dir / "isc_sig_fdrq05.nii.gz",
    }
    nib.save(mean_img, str(paths["isc_mean"]))
    nib.save(p_img, str(paths["isc_p"]))
    nib.save(q_img, str(paths["isc_q"]))
    nib.save(sig_img, str(paths["isc_sig"]))
    return {k: str(v) for k, v in paths.items()}
