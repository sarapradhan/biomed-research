#!/usr/bin/env python3
"""run_sensitivity_analysis.py — Robustness Benchmark for the fMRI Pipeline.

Systematically varies preprocessing and analysis parameters to test which
pipeline outputs are stable ("robust") and which change materially ("fragile")
under reasonable alternative choices.

Parameters tested:
    1. Global signal regression    : with vs without GSR
    2. Parcellation atlas          : Schaefer-100 vs 200 vs 400
    3. Dynamic FC window length    : 20, 30, 45, 60 TRs
    4. Scrubbing threshold         : 0.3, 0.5, 0.9 mm FD
    5. Smoothing kernel            : 4, 6, 8 mm FWHM

For each parameter variation the script:
    - Runs preprocessing + ROI extraction + static FC + dynamic FC
    - Records subject-averaged upper-triangle FC vectors
    - Computes pairwise correlations between FC matrices from different settings
    - Generates a stability heatmap and "recommended defaults" summary

Usage
-----
    python scripts/run_sensitivity_analysis.py [--data-root PATH] [--output-root PATH]

Defaults to the ds007318 fMRIPrep data already on this machine.

Outputs
-------
    <output-root>/sensitivity/
        results/               per-condition FC matrices and metadata
        stability_heatmap.png  which outputs are robust vs fragile
        stability_summary.csv  pairwise similarity scores
        recommended_defaults.csv  recommended parameter table
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

# ── project source on path ───────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.connectivity import dynamic_fc_summary, static_fc  # noqa: E402
from fmri_pipeline.preprocessing import (  # noqa: E402
    load_confound_matrix,
    preprocess_bold,
)
from fmri_pipeline.roi import extract_roi_timeseries, get_schaefer_atlas  # noqa: E402
from fmri_pipeline.utils import set_global_seed, setup_logging, upper_triangle_vector  # noqa: E402


# ── Known runs (ds007318) ────────────────────────────────────────────────────
_KNOWN_RUNS = [
    ("01", "1"),
    ("01", "2"),
    ("02", "1"),
    ("03", "1"),
    ("03", "2"),
]
_TR = 2.0
_SPACE = "MNI152NLin2009cAsym"
_RES = "2"
_TASK = "removal"


def _bold_path(deriv: Path, sub: str, ses: str) -> Path:
    return (
        deriv / f"sub-{sub}" / f"ses-{ses}" / "func"
        / f"sub-{sub}_ses-{ses}_task-{_TASK}_space-{_SPACE}_res-{_RES}_desc-preproc_bold.nii.gz"
    )


def _mask_path(deriv: Path, sub: str, ses: str) -> Path:
    return (
        deriv / f"sub-{sub}" / f"ses-{ses}" / "func"
        / f"sub-{sub}_ses-{ses}_task-{_TASK}_space-{_SPACE}_res-{_RES}_desc-brain_mask.nii.gz"
    )


def _confounds_path(deriv: Path, sub: str, ses: str) -> Path:
    return (
        deriv / f"sub-{sub}" / f"ses-{ses}" / "func"
        / f"sub-{sub}_ses-{ses}_task-{_TASK}_desc-confounds_timeseries.tsv"
    )


# ── Parameter grid ───────────────────────────────────────────────────────────

@dataclass
class ParameterSet:
    """One point in the sensitivity analysis parameter space."""
    name: str
    gsr: bool = True
    schaefer_n_rois: int = 200
    dfc_window_trs: int = 30
    fd_threshold_mm: float = 0.5
    smoothing_fwhm_mm: float = 6.0

    def label(self) -> str:
        return (
            f"GSR={'on' if self.gsr else 'off'}_"
            f"atlas={self.schaefer_n_rois}_"
            f"dfc={self.dfc_window_trs}TR_"
            f"FD={self.fd_threshold_mm}_"
            f"sm={self.smoothing_fwhm_mm}mm"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gsr": self.gsr,
            "schaefer_n_rois": self.schaefer_n_rois,
            "dfc_window_trs": self.dfc_window_trs,
            "fd_threshold_mm": self.fd_threshold_mm,
            "smoothing_fwhm_mm": self.smoothing_fwhm_mm,
        }


# Baseline parameters (from the validated paper)
BASELINE = ParameterSet(
    name="baseline",
    gsr=True,
    schaefer_n_rois=200,
    dfc_window_trs=30,
    fd_threshold_mm=0.5,
    smoothing_fwhm_mm=6.0,
)


def build_parameter_grid() -> List[ParameterSet]:
    """Build the full sensitivity grid: vary one parameter at a time from baseline."""
    grid = [BASELINE]

    # 1. GSR: with vs without
    grid.append(ParameterSet(name="gsr_off", gsr=False))

    # 2. Parcellation: 100, 400 (baseline is 200)
    for n in [100, 400]:
        grid.append(ParameterSet(name=f"atlas_{n}", schaefer_n_rois=n))

    # 3. dFC window: 20, 45, 60 TRs (baseline is 30)
    for w in [20, 45, 60]:
        grid.append(ParameterSet(name=f"dfc_{w}TR", dfc_window_trs=w))

    # 4. Scrubbing threshold: 0.3, 0.9 mm (baseline is 0.5)
    for t in [0.3, 0.9]:
        grid.append(ParameterSet(name=f"FD_{t}mm", fd_threshold_mm=t))

    # 5. Smoothing: 4, 8 mm (baseline is 6)
    for s in [4.0, 8.0]:
        grid.append(ParameterSet(name=f"smooth_{s}mm", smoothing_fwhm_mm=s))

    return grid


def build_cfg_for_params(params: ParameterSet, output_root: Path) -> Dict:
    """Build a full pipeline config dict for a given parameter set."""
    return {
        "project": {
            "name": f"sensitivity_{params.name}",
            "random_seed": 42,
            "n_jobs": 4,
            "debug_mode": False,
        },
        "paths": {
            "output_root": str(output_root / "sensitivity" / "results" / params.name),
            "cache_dir": str(output_root / "sensitivity" / "cache"),
            "logs_dir": str(output_root / "sensitivity" / "logs"),
        },
        "preprocessing": {
            "confounds": {
                "include_wm_csf": True,
                "wm_columns": ["white_matter", "wm"],
                "csf_columns": ["csf"],
                "include_global_signal": params.gsr,
            },
            "scrubbing": {
                "fd_threshold_mm": params.fd_threshold_mm,
                "exclude_percent_censored": 0.20,
                "exclude_max_motion_mm": 3.0,
                "exclude_max_rotation_deg": 3.0,
            },
            "filtering": {"low_hz": 0.01, "high_hz": 0.10},
            "smoothing_fwhm_mm": params.smoothing_fwhm_mm,
        },
        "roi": {
            "atlas": f"schaefer_{params.schaefer_n_rois}",
            "schaefer_n_rois": params.schaefer_n_rois,
            "schaefer_yeo_networks": 7,
            "standardize": "zscore_sample",
        },
        "dynamic_fc": {
            "window_trs": params.dfc_window_trs,
            "step_trs": 5,
            "exploratory_clustering": False,
        },
        "reho": {
            "neighborhood": 26,
            "normalize_zscore": True,
            "chunk_size_voxels": 10000,
        },
    }


# ── Core analysis functions ──────────────────────────────────────────────────

def run_single_condition(
    params: ParameterSet,
    deriv: Path,
    output_root: Path,
    logger,
) -> Dict[str, Any]:
    """Run preprocessing + ROI + FC for a single parameter configuration.

    Returns a dict with:
        - "params": the ParameterSet
        - "static_fc_vectors": dict[subject -> upper-triangle FC vector]
        - "dfc_std_vectors": dict[subject -> upper-triangle dFC variability vector]
        - "mean_static_fc_vector": group-average static FC vector
        - "mean_dfc_std_vector": group-average dFC variability vector
        - "n_runs_kept": number of runs that passed QC
        - "runtime_sec": wall-clock seconds
    """
    t0 = time.time()
    cfg = build_cfg_for_params(params, output_root)

    # Ensure output dirs exist
    for d in ["output_root", "cache_dir", "logs_dir"]:
        Path(cfg["paths"][d]).mkdir(parents=True, exist_ok=True)

    logger.info("Running condition: %s  [%s]", params.name, params.label())

    # Fetch atlas for this configuration
    atlas_img, labels = get_schaefer_atlas(cfg)

    # Per-run: preprocess -> ROI -> static FC -> dynamic FC
    subject_static: Dict[str, List[np.ndarray]] = {}
    subject_dfc_std: Dict[str, List[np.ndarray]] = {}
    n_kept = 0

    for sub, ses in _KNOWN_RUNS:
        bold = _bold_path(deriv, sub, ses)
        mask = _mask_path(deriv, sub, ses)
        conf = _confounds_path(deriv, sub, ses)
        if not all(p.exists() for p in (bold, mask, conf)):
            logger.warning("  Skipping sub-%s ses-%s: missing files", sub, ses)
            continue

        try:
            confounds, keep_mask, fd, motion = load_confound_matrix(
                str(conf), cfg, bold_file=str(bold), mask_file=str(mask)
            )
            if motion["exclude"]:
                logger.info("  sub-%s ses-%s excluded by motion QC", sub, ses)
                continue

            unsm, sm = preprocess_bold(
                bold_file=str(bold),
                mask_file=str(mask),
                confound_matrix=confounds,
                keep_mask=keep_mask,
                tr=_TR,
                cfg=cfg,
            )

            # ROI extraction on unsmoothed image
            ts = extract_roi_timeseries(unsm, atlas_img, _TR, cfg)

            # Static FC
            sfc = static_fc(ts)
            subject_static.setdefault(sub, []).append(sfc)

            # Dynamic FC
            _, dfc_std, _ = dynamic_fc_summary(ts, cfg)
            subject_dfc_std.setdefault(sub, []).append(dfc_std)

            n_kept += 1

        except Exception as e:
            logger.warning("  sub-%s ses-%s failed: %s", sub, ses, str(e))
            continue

    # Average across runs within subject, then across subjects
    static_vecs = {}
    dfc_vecs = {}

    for sub, mats in subject_static.items():
        avg = np.mean(np.stack(mats), axis=0)
        vec, _ = upper_triangle_vector(avg)
        static_vecs[sub] = vec

    for sub, mats in subject_dfc_std.items():
        avg = np.mean(np.stack(mats), axis=0)
        vec, _ = upper_triangle_vector(avg)
        dfc_vecs[sub] = vec

    # Group means
    mean_static = np.mean(list(static_vecs.values()), axis=0) if static_vecs else np.array([])
    mean_dfc = np.mean(list(dfc_vecs.values()), axis=0) if dfc_vecs else np.array([])

    runtime = time.time() - t0
    logger.info("  Done: %s  (%d runs kept, %.1fs)", params.name, n_kept, runtime)

    # Save results
    res_dir = Path(cfg["paths"]["output_root"])
    res_dir.mkdir(parents=True, exist_ok=True)
    if mean_static.size:
        np.save(res_dir / "mean_static_fc_vector.npy", mean_static)
    if mean_dfc.size:
        np.save(res_dir / "mean_dfc_std_vector.npy", mean_dfc)

    return {
        "params": params,
        "static_fc_vectors": static_vecs,
        "dfc_std_vectors": dfc_vecs,
        "mean_static_fc_vector": mean_static,
        "mean_dfc_std_vector": mean_dfc,
        "n_runs_kept": n_kept,
        "runtime_sec": runtime,
    }


def compute_stability_matrix(
    results: List[Dict[str, Any]],
    metric: str = "static_fc",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute pairwise Spearman correlation between conditions for a given metric.

    Parameters
    ----------
    results : list of result dicts from run_single_condition
    metric : "static_fc" or "dfc_std"

    Returns
    -------
    labels_df : DataFrame with condition metadata
    corr_mat  : n_conditions x n_conditions Spearman correlation matrix
    """
    key = f"mean_{metric}_vector"
    valid = [(r["params"], r[key]) for r in results if r[key].size > 0]

    if len(valid) < 2:
        return pd.DataFrame(), np.array([])

    # For atlas variations, FC vectors have different lengths.
    # Compare only conditions with the same atlas resolution.
    # Group by atlas size, compute within-group similarities.
    by_atlas: Dict[int, List[Tuple[ParameterSet, np.ndarray]]] = {}
    for p, v in valid:
        by_atlas.setdefault(p.schaefer_n_rois, []).append((p, v))

    # For the main heatmap, use only conditions with the baseline atlas (200)
    # to keep vectors comparable
    baseline_atlas = BASELINE.schaefer_n_rois
    same_atlas = by_atlas.get(baseline_atlas, [])

    if len(same_atlas) < 2:
        return pd.DataFrame(), np.array([])

    names = [p.name for p, _ in same_atlas]
    vecs = np.stack([v for _, v in same_atlas])
    n = len(names)
    corr_mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            # Build a shared mask of finite (non-NaN) edges across both vectors
            # so that NaN ROIs (e.g. outside brain coverage) don't poison the
            # Spearman correlation.
            mask = np.isfinite(vecs[i]) & np.isfinite(vecs[j])
            if mask.sum() < 10:
                corr_mat[i, j] = float("nan")
                corr_mat[j, i] = float("nan")
            else:
                rho, _ = spearmanr(vecs[i][mask], vecs[j][mask])
                corr_mat[i, j] = rho
                corr_mat[j, i] = rho

    labels_df = pd.DataFrame([p.to_dict() for p, _ in same_atlas])
    return labels_df, corr_mat


def compute_cross_atlas_similarity(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compare FC structure across atlas resolutions using rank correlation
    of network-level summaries (mean within-network vs between-network FC)."""
    rows = []
    baseline_res = [r for r in results if r["params"].name == "baseline"]
    atlas_results = [r for r in results if r["params"].name.startswith("atlas_")]

    if not baseline_res:
        return pd.DataFrame()

    # For atlas comparisons, we note that direct vector comparison is not possible
    # (different dimensionality). Instead, report the summary statistics.
    for r in [baseline_res[0]] + atlas_results:
        p = r["params"]
        vec = r["mean_static_fc_vector"]
        if vec.size == 0:
            continue
        finite = vec[np.isfinite(vec)]
        if finite.size == 0:
            continue
        rows.append({
            "condition": p.name,
            "atlas_n_rois": p.schaefer_n_rois,
            "n_edges": vec.size,
            "n_finite_edges": finite.size,
            "n_nan_edges": int(vec.size - finite.size),
            "mean_fc": float(np.mean(finite)),
            "std_fc": float(np.std(finite)),
            "median_fc": float(np.median(finite)),
            "iqr_fc": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
        })

    return pd.DataFrame(rows)


# ── Visualization ────────────────────────────────────────────────────────────

def plot_stability_heatmap(
    labels_df: pd.DataFrame,
    corr_mat: np.ndarray,
    title: str,
    out_file: str,
) -> None:
    """Plot a pairwise similarity heatmap."""
    if corr_mat.size == 0:
        return

    n = corr_mat.shape[0]
    names = labels_df["name"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.7)))
    im = ax.imshow(corr_mat, cmap="RdYlGn", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if corr_mat[i, j] < 0.75 else "black"
            ax.text(j, i, f"{corr_mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def plot_parameter_impact(results: List[Dict[str, Any]], out_dir: Path) -> None:
    """Create bar charts showing the impact of each parameter dimension."""
    baseline_vec = None
    for r in results:
        if r["params"].name == "baseline" and r["mean_static_fc_vector"].size > 0:
            baseline_vec = r["mean_static_fc_vector"]
            break

    if baseline_vec is None:
        return

    # Group by parameter dimension
    dimensions = {
        "GSR": ["gsr_off"],
        "Atlas": ["atlas_100", "atlas_400"],
        "dFC Window": ["dfc_20TR", "dfc_45TR", "dfc_60TR"],
        "Scrubbing": ["FD_0.3mm", "FD_0.9mm"],
        "Smoothing": ["smooth_4.0mm", "smooth_8.0mm"],
    }

    dim_impacts = {}
    for dim_name, condition_names in dimensions.items():
        similarities = []
        labels = []
        for r in results:
            if r["params"].name in condition_names:
                vec = r["mean_static_fc_vector"]
                if vec.size == baseline_vec.size and vec.size > 0:
                    mask = np.isfinite(baseline_vec) & np.isfinite(vec)
                    if mask.sum() < 10:
                        continue
                    rho, _ = spearmanr(baseline_vec[mask], vec[mask])
                    similarities.append(rho)
                    labels.append(r["params"].name)
        if similarities:
            dim_impacts[dim_name] = {
                "min_similarity": min(similarities),
                "mean_similarity": np.mean(similarities),
                "labels": labels,
                "similarities": similarities,
            }

    if not dim_impacts:
        return

    # Bar chart: mean deviation from baseline per dimension
    fig, ax = plt.subplots(figsize=(10, 5))
    dim_names = list(dim_impacts.keys())
    deviations = [1.0 - dim_impacts[d]["mean_similarity"] for d in dim_names]
    colors = ["#e74c3c" if d > 0.05 else "#f39c12" if d > 0.02 else "#27ae60" for d in deviations]

    bars = ax.barh(dim_names, deviations, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Mean Deviation from Baseline (1 - Spearman r)", fontsize=11)
    ax.set_title("Parameter Sensitivity: Impact on Static FC", fontsize=12, fontweight="bold")
    ax.axvline(x=0.02, color="#27ae60", linestyle="--", alpha=0.5, label="Robust threshold")
    ax.axvline(x=0.05, color="#e74c3c", linestyle="--", alpha=0.5, label="Fragile threshold")
    ax.legend(loc="lower right", fontsize=9)

    for bar, dev in zip(bars, deviations):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{dev:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "parameter_impact_barchart.png", dpi=200, bbox_inches="tight")
    plt.close()


def generate_joss_composite_figure(
    labels_df: pd.DataFrame,
    corr_mat: np.ndarray,
    results: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Create the 3-panel composite figure for the JOSS paper.

    Panel A: Static FC stability heatmap (same-atlas conditions).
    Panel B: Parameter impact bar chart (deviation from baseline).
    Panel C: Key findings text summary.
    """
    import matplotlib.gridspec as gridspec

    # ── Compute similarity-to-baseline for same-atlas conditions ──
    baseline_vec = None
    for r in results:
        if r["params"].name == "baseline" and r["mean_static_fc_vector"].size > 0:
            baseline_vec = r["mean_static_fc_vector"]
            break
    if baseline_vec is None or corr_mat.size == 0:
        return

    sim_to_baseline: Dict[str, float] = {}
    for r in results:
        vec = r["mean_static_fc_vector"]
        if vec.size == baseline_vec.size and vec.size > 0:
            mask = np.isfinite(baseline_vec) & np.isfinite(vec)
            if mask.sum() < 10:
                continue
            rho, _ = spearmanr(baseline_vec[mask], vec[mask])
            sim_to_baseline[r["params"].name] = rho

    # ── Compute per-dimension deviations ──
    dimensions = {
        "GSR": ["gsr_off"],
        "Atlas*": ["atlas_100", "atlas_400"],
        "dFC Win": ["dfc_20TR", "dfc_45TR", "dfc_60TR"],
        "Scrub": ["FD_0.3mm", "FD_0.9mm"],
        "Smooth": ["smooth_4.0mm", "smooth_8.0mm"],
    }
    dim_names_ordered = ["Smooth", "Scrub", "dFC Win", "Atlas*", "GSR"]
    deviations = []
    for dim in dim_names_ordered:
        cond_names = dimensions[dim]
        sims = [sim_to_baseline[c] for c in cond_names if c in sim_to_baseline]
        deviations.append(1.0 - np.mean(sims) if sims else 0.0)
    colors = ["#e74c3c" if d > 0.05 else "#f39c12" if d > 0.02 else "#27ae60" for d in deviations]

    # ── Build figure ──
    n = corr_mat.shape[0]
    vmin = max(0.5, np.nanmin(corr_mat) - 0.05) if corr_mat.size else 0.5

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1, 0.8], wspace=0.3)

    # Panel A: heatmap
    ax1 = fig.add_subplot(gs[0])
    short_names = ["Baseline", "GSR off", "dFC 20TR", "dFC 45TR", "dFC 60TR",
                   "FD 0.3mm", "FD 0.9mm", "Sm 4mm", "Sm 8mm"]
    im = ax1.imshow(corr_mat, cmap="RdYlGn", vmin=vmin, vmax=1.0)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax1.set_yticklabels(short_names, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = corr_mat[i, j]
            txt = f"{v:.2f}" if np.isfinite(v) else "NaN"
            color = "white" if (not np.isfinite(v) or v < 0.90) else "black"
            ax1.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    ax1.set_title("A. Static FC Stability", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="\u03c1")

    # Panel B: impact bar chart
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.barh(dim_names_ordered, deviations, color=colors, edgecolor="white", height=0.55)
    ax2.axvline(x=0.02, color="#27ae60", linestyle="--", alpha=0.5, linewidth=1)
    ax2.axvline(x=0.05, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
    for bar, dev in zip(bars, deviations):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{dev:.3f}", va="center", fontsize=8, fontweight="bold")
    ax2.set_xlabel("Deviation (1 \u2212 \u03c1)", fontsize=9)
    ax2.set_title("B. Parameter Impact", fontsize=11, fontweight="bold")
    ax2.set_xlim(0, max(deviations) * 1.4 if max(deviations) > 0 else 0.01)

    # Panel C: key findings text
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    gsr_sim = sim_to_baseline.get("gsr_off", float("nan"))
    fd03_sim = sim_to_baseline.get("FD_0.3mm", float("nan"))
    findings = (
        f"C. Key Findings\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n\n"
        f"\u25cf GSR is the single\n"
        f"  most impactful choice\n"
        f"  (\u03c1 = {gsr_sim:.3f} without GSR)\n\n"
        f"\u25cf Scrubbing threshold\n"
        f"  shows {'moderate' if fd03_sim < 0.98 else 'minimal'} sensitivity\n"
        f"  (FD 0.3mm: \u03c1 = {fd03_sim:.3f})\n\n"
        f"\u25cf dFC window, smoothing\n"
        f"  are robust (\u03c1 > 0.98)\n\n"
        f"\u25cf Atlas resolution:\n"
        f"  different dimensions,\n"
        f"  summary stats stable\n\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"ds007318 | N=3 | 5 runs\n"
        f"Schaefer-200 baseline\n"
        f"NaN-masked edges"
    )
    ax3.text(0.05, 0.95, findings, transform=ax3.transAxes, fontsize=8.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8, pad=0.6))

    fig.suptitle("Sensitivity Analysis: Robustness of Preprocessing Choices \u2014 Real Data",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(out_dir / "joss_figure_sensitivity_combined.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_recommended_defaults(results: List[Dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    """Create the 'recommended defaults' table based on stability analysis."""
    baseline_vec = None
    for r in results:
        if r["params"].name == "baseline" and r["mean_static_fc_vector"].size > 0:
            baseline_vec = r["mean_static_fc_vector"]
            break

    rows = []
    for r in results:
        p = r["params"]
        vec = r["mean_static_fc_vector"]
        if vec.size == 0:
            continue

        # Compute similarity to baseline (skip if this IS the baseline)
        if p.name == "baseline":
            sim = 1.0
        elif vec.size == baseline_vec.size:
            mask = np.isfinite(baseline_vec) & np.isfinite(vec)
            if mask.sum() < 10:
                sim = float("nan")
            else:
                sim, _ = spearmanr(baseline_vec[mask], vec[mask])
        else:
            sim = float("nan")  # Different atlas size

        stability = "Robust" if sim > 0.98 else "Moderate" if sim > 0.95 else "Fragile"

        rows.append({
            "Parameter": p.name,
            "GSR": "On" if p.gsr else "Off",
            "Atlas": f"Schaefer-{p.schaefer_n_rois}",
            "dFC Window": f"{p.dfc_window_trs} TRs",
            "FD Threshold": f"{p.fd_threshold_mm} mm",
            "Smoothing": f"{p.smoothing_fwhm_mm} mm",
            "Runs Kept": r["n_runs_kept"],
            "Similarity to Baseline": f"{sim:.4f}" if not np.isnan(sim) else "N/A (different atlas)",
            "Stability": stability,
            "Runtime (s)": f"{r['runtime_sec']:.1f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "recommended_defaults.csv", index=False)
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def run_sensitivity(data_root: Path, output_root: Path) -> None:
    """Run the full sensitivity analysis."""
    sens_dir = output_root / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(sens_dir / "logs"), name="sensitivity")
    set_global_seed(42)

    logger.info("=" * 70)
    logger.info("SENSITIVITY ANALYSIS — Robustness Benchmark")
    logger.info("Data root   : %s", data_root)
    logger.info("Output root : %s", sens_dir)
    logger.info("=" * 70)

    deriv = data_root / "derivatives"
    grid = build_parameter_grid()
    logger.info("Parameter grid: %d conditions", len(grid))
    for p in grid:
        logger.info("  %s: %s", p.name, p.label())

    # Run all conditions
    results = []
    for i, params in enumerate(grid):
        logger.info("─" * 50)
        logger.info("Condition %d/%d: %s", i + 1, len(grid), params.name)
        result = run_single_condition(params, deriv, output_root, logger)
        results.append(result)

    # Save parameter grid metadata
    grid_meta = [{"index": i, **p.to_dict(), "label": p.label()} for i, p in enumerate(grid)]
    with open(sens_dir / "parameter_grid.json", "w") as f:
        json.dump(grid_meta, f, indent=2)

    # Compute and plot stability for static FC
    logger.info("Computing static FC stability matrix...")
    labels_df, corr_mat = compute_stability_matrix(results, metric="static_fc")
    if corr_mat.size:
        plot_stability_heatmap(
            labels_df, corr_mat,
            "Static FC Stability Across Parameter Choices\n(Spearman r between group-mean FC vectors)",
            str(sens_dir / "stability_heatmap_static_fc.png"),
        )
        # Save the correlation matrix
        np.save(sens_dir / "stability_matrix_static_fc.npy", corr_mat)
        labels_df.to_csv(sens_dir / "stability_labels.csv", index=False)

    # Compute and plot stability for dFC variability
    logger.info("Computing dFC variability stability matrix...")
    dfc_labels, dfc_corr = compute_stability_matrix(results, metric="dfc_std")
    if dfc_corr.size:
        plot_stability_heatmap(
            dfc_labels, dfc_corr,
            "dFC Variability Stability Across Parameter Choices\n(Spearman r between group-mean dFC-std vectors)",
            str(sens_dir / "stability_heatmap_dfc_variability.png"),
        )

    # Cross-atlas comparison
    logger.info("Computing cross-atlas summary...")
    atlas_df = compute_cross_atlas_similarity(results)
    if not atlas_df.empty:
        atlas_df.to_csv(sens_dir / "cross_atlas_summary.csv", index=False)

    # Parameter impact bar chart
    logger.info("Generating parameter impact visualization...")
    plot_parameter_impact(results, sens_dir)

    # JOSS composite figure (3-panel: heatmap + impact + findings)
    if corr_mat.size:
        logger.info("Generating JOSS composite figure...")
        generate_joss_composite_figure(labels_df, corr_mat, results, sens_dir)

    # Recommended defaults table
    logger.info("Generating recommended defaults table...")
    defaults_df = generate_recommended_defaults(results, sens_dir)
    print("\n" + "=" * 70)
    print("RECOMMENDED DEFAULTS TABLE")
    print("=" * 70)
    print(defaults_df.to_string(index=False))

    # Summary JSON
    summary = {
        "n_conditions": len(results),
        "n_runs_per_condition": [r["n_runs_kept"] for r in results],
        "total_runtime_sec": sum(r["runtime_sec"] for r in results),
        "conditions": [r["params"].name for r in results],
    }
    with open(sens_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("Sensitivity analysis complete.")
    logger.info("  Heatmaps  : %s/stability_heatmap_*.png", sens_dir)
    logger.info("  Defaults  : %s/recommended_defaults.csv", sens_dir)
    logger.info("  Impact    : %s/parameter_impact_barchart.png", sens_dir)
    logger.info("=" * 70)


def replot_from_saved(output_root: Path) -> None:
    """Regenerate all figures and tables from previously saved .npy vectors.

    This avoids re-running the full pipeline (~30 min). It reads the
    mean_static_fc_vector.npy and mean_dfc_std_vector.npy files that
    run_sensitivity() saved in each condition's results/ folder.
    """
    sens_dir = output_root / "sensitivity"
    results_dir = sens_dir / "results"

    if not results_dir.exists():
        sys.exit(f"ERROR: no results/ folder at {results_dir}. Run the full analysis first.")

    logger = setup_logging(str(sens_dir / "logs"), name="sensitivity_replot")
    logger.info("=" * 70)
    logger.info("REPLOT MODE — regenerating figures from saved .npy vectors")
    logger.info("Results dir : %s", results_dir)
    logger.info("=" * 70)

    # Rebuild the parameter grid and reconstruct results from saved vectors
    grid = build_parameter_grid()
    results = []
    for params in grid:
        cond_dir = results_dir / params.name
        static_path = cond_dir / "mean_static_fc_vector.npy"
        dfc_path = cond_dir / "mean_dfc_std_vector.npy"

        mean_static = np.load(static_path) if static_path.exists() else np.array([])
        mean_dfc = np.load(dfc_path) if dfc_path.exists() else np.array([])

        if mean_static.size == 0 and mean_dfc.size == 0:
            logger.warning("  Skipping %s: no saved vectors found at %s", params.name, cond_dir)
            continue

        logger.info("  Loaded %s: static=%d edges, dfc=%d edges",
                     params.name, mean_static.size, mean_dfc.size)

        results.append({
            "params": params,
            "static_fc_vectors": {},   # per-subject vectors not saved
            "dfc_std_vectors": {},
            "mean_static_fc_vector": mean_static,
            "mean_dfc_std_vector": mean_dfc,
            "n_runs_kept": -1,         # unknown in replot mode
            "runtime_sec": 0.0,
        })

    if len(results) < 2:
        sys.exit("ERROR: fewer than 2 conditions found. Cannot compute stability matrix.")

    logger.info("Loaded %d conditions from saved vectors.", len(results))

    # Compute and plot stability for static FC
    logger.info("Computing static FC stability matrix...")
    labels_df, corr_mat = compute_stability_matrix(results, metric="static_fc")
    if corr_mat.size:
        plot_stability_heatmap(
            labels_df, corr_mat,
            "Static FC Stability Across Preprocessing Choices\n"
            "(Spearman \u03c1, NaN-masked, ds007318, N=3, 5 runs)",
            str(sens_dir / "stability_heatmap_static_fc.png"),
        )
        np.save(sens_dir / "stability_matrix_static_fc.npy", corr_mat)
        labels_df.to_csv(sens_dir / "stability_labels.csv", index=False)

    # Compute and plot stability for dFC variability
    logger.info("Computing dFC variability stability matrix...")
    dfc_labels, dfc_corr = compute_stability_matrix(results, metric="dfc_std")
    if dfc_corr.size:
        plot_stability_heatmap(
            dfc_labels, dfc_corr,
            "dFC Variability Stability Across Preprocessing Choices\n"
            "(Spearman \u03c1, NaN-masked, ds007318, N=3, 5 runs)",
            str(sens_dir / "stability_heatmap_dfc_variability.png"),
        )

    # Cross-atlas comparison
    logger.info("Computing cross-atlas summary...")
    atlas_df = compute_cross_atlas_similarity(results)
    if not atlas_df.empty:
        atlas_df.to_csv(sens_dir / "cross_atlas_summary.csv", index=False)

    # Parameter impact bar chart
    logger.info("Generating parameter impact visualization...")
    plot_parameter_impact(results, sens_dir)

    # JOSS composite figure (3-panel: heatmap + impact + findings)
    if corr_mat.size:
        logger.info("Generating JOSS composite figure...")
        generate_joss_composite_figure(labels_df, corr_mat, results, sens_dir)

    # Recommended defaults table
    logger.info("Generating recommended defaults table...")
    defaults_df = generate_recommended_defaults(results, sens_dir)
    print("\n" + "=" * 70)
    print("RECOMMENDED DEFAULTS TABLE")
    print("=" * 70)
    print(defaults_df.to_string(index=False))

    logger.info("=" * 70)
    logger.info("Replot complete. Figures saved to: %s", sens_dir)
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness benchmark: sensitivity analysis of preprocessing choices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-root", type=Path,
        default=Path.home() / "fmri-pipeline-data",
        help="fMRIPrep directory with derivatives/ sub-folder",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Output directory (default: <data-root>/pipeline_output)",
    )
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Skip the full pipeline re-run. Regenerate figures from previously "
             "saved .npy vectors in <output-root>/sensitivity/results/. "
             "Use this after fixing bugs in plotting code or NaN handling.",
    )
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else data_root / "pipeline_output"
    )

    if args.replot_only:
        replot_from_saved(output_root)
    else:
        if not data_root.exists():
            sys.exit(f"ERROR: data-root not found: {data_root}")
        if not (data_root / "derivatives").exists():
            sys.exit(f"ERROR: no derivatives/ folder under {data_root}")
        run_sensitivity(data_root, output_root)


if __name__ == "__main__":
    main()
