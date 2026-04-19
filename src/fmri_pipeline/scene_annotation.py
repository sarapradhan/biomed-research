"""scene_annotation.py — Scene annotation framework and ISC-to-scene alignment.

This module supports the Track 2 ISC extension by providing tools to:
    1. Load and validate scene annotation CSVs
    2. Convert scene timing (seconds) to fMRI volume indices
    3. Align ISC maps with scene features (emotional valence, social cognition,
       narrative transitions) to test whether intersubject synchrony covaries
       with stimulus content

Scene Annotation Format
-----------------------
Each episode's annotation is a CSV with at minimum these columns:
    onset_sec       float   Scene onset in seconds from stimulus start
    offset_sec      float   Scene offset in seconds from stimulus start

Plus one or more feature columns (configurable):
    emotional_valence      float   -1 (negative) to +1 (positive)
    social_cognition       int     0 = no theory-of-mind, 1 = ToM moment
    narrative_transition   int     0 = within-scene, 1 = scene boundary

Example CSV
-----------
    onset_sec,offset_sec,emotional_valence,social_cognition,narrative_transition
    0.0,45.2,0.3,0,0
    45.2,78.6,-0.5,1,1
    78.6,120.0,0.8,0,0

Usage
-----
    from fmri_pipeline.scene_annotation import (
        load_scene_annotations,
        scenes_to_tr_indices,
        align_isc_to_scenes,
        correlate_isc_with_features,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ── Loading and validation ───────────────────────────────────────────────────

def load_scene_annotations(
    annotation_path: str,
    required_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load a scene annotation CSV and validate its structure.

    Parameters
    ----------
    annotation_path : str
        Path to the scene annotation CSV file.
    required_columns : list of str, optional
        Feature columns that must be present. Defaults to the three core features.

    Returns
    -------
    pd.DataFrame
        Validated annotation DataFrame sorted by onset_sec.

    Raises
    ------
    ValueError
        If required columns are missing or timing is invalid.
    """
    if required_columns is None:
        required_columns = ["emotional_valence", "social_cognition", "narrative_transition"]

    df = pd.read_csv(annotation_path)

    # Check required timing columns
    for col in ["onset_sec", "offset_sec"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {annotation_path}")

    # Check at least one feature column
    available_features = [c for c in required_columns if c in df.columns]
    if not available_features:
        raise ValueError(
            f"No feature columns found in {annotation_path}. "
            f"Expected at least one of: {required_columns}"
        )

    # Validate timing
    df = df.sort_values("onset_sec").reset_index(drop=True)
    if (df["offset_sec"] <= df["onset_sec"]).any():
        raise ValueError(f"Invalid timing: offset_sec must be > onset_sec in {annotation_path}")

    return df


def load_all_annotations(
    annotation_dir: str,
    pattern: str = "*.csv",
    required_columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load all scene annotation CSVs from a directory.

    Returns a dict mapping episode name (filename stem) to DataFrame.
    """
    ann_dir = Path(annotation_dir)
    if not ann_dir.exists():
        return {}

    annotations = {}
    for csv_file in sorted(ann_dir.glob(pattern)):
        try:
            df = load_scene_annotations(str(csv_file), required_columns)
            annotations[csv_file.stem] = df
        except (ValueError, pd.errors.ParserError) as e:
            print(f"  Warning: skipping {csv_file.name}: {e}")

    return annotations


# ── Timing conversion ────────────────────────────────────────────────────────

def scenes_to_tr_indices(
    scenes_df: pd.DataFrame,
    tr_sec: float,
    n_volumes: Optional[int] = None,
) -> pd.DataFrame:
    """Convert scene onset/offset from seconds to TR (volume) indices.

    Parameters
    ----------
    scenes_df : pd.DataFrame
        Scene annotation with onset_sec and offset_sec columns.
    tr_sec : float
        Repetition time in seconds.
    n_volumes : int, optional
        Total number of volumes in the scan. If provided, clips indices.

    Returns
    -------
    pd.DataFrame
        Copy of scenes_df with added columns:
        onset_tr, offset_tr, duration_trs
    """
    df = scenes_df.copy()
    df["onset_tr"] = np.floor(df["onset_sec"] / tr_sec).astype(int)
    df["offset_tr"] = np.ceil(df["offset_sec"] / tr_sec).astype(int)

    if n_volumes is not None:
        df["offset_tr"] = df["offset_tr"].clip(upper=n_volumes)

    df["duration_trs"] = df["offset_tr"] - df["onset_tr"]
    return df


# ── ISC-to-scene alignment ──────────────────────────────────────────────────

def compute_temporal_isc_profile(
    isc_subject_maps: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a temporal ISC profile from per-volume or per-window ISC data.

    If isc_subject_maps is shape (n_subjects, n_voxels) from the leave-one-out
    ISC, returns the mean ISC across subjects.

    If roi_mask is provided, averages only within the mask.

    Parameters
    ----------
    isc_subject_maps : np.ndarray
        ISC data. Shape depends on computation method.
    roi_mask : np.ndarray, optional
        Boolean mask for spatial averaging.

    Returns
    -------
    np.ndarray
        Temporal ISC profile (1D array, one value per time point or window).
    """
    if roi_mask is not None:
        masked = isc_subject_maps[:, roi_mask]
    else:
        masked = isc_subject_maps

    return np.mean(masked, axis=-1)


def align_isc_to_scenes(
    isc_timecourse: np.ndarray,
    scenes_df: pd.DataFrame,
    summary_method: str = "mean",
    min_scene_trs: int = 5,
    zscore_isc: bool = True,
) -> pd.DataFrame:
    """Align a temporal ISC profile to scene segments.

    For each scene in scenes_df (which must have onset_tr and offset_tr columns),
    compute the summary ISC within that scene's time window.

    Parameters
    ----------
    isc_timecourse : np.ndarray
        1D array of ISC values (one per TR or per window).
    scenes_df : pd.DataFrame
        Scene annotation with onset_tr and offset_tr columns.
    summary_method : str
        "mean" or "median" — how to summarize ISC within each scene.
    min_scene_trs : int
        Minimum scene duration in TRs to include.
    zscore_isc : bool
        Whether to z-score ISC values before alignment.

    Returns
    -------
    pd.DataFrame
        scenes_df with an added 'scene_isc' column.
    """
    df = scenes_df.copy()

    if zscore_isc and np.std(isc_timecourse) > 0:
        isc_z = (isc_timecourse - np.mean(isc_timecourse)) / np.std(isc_timecourse)
    else:
        isc_z = isc_timecourse

    scene_isc_values = []
    for _, row in df.iterrows():
        onset = int(row["onset_tr"])
        offset = int(row["offset_tr"])
        duration = offset - onset

        if duration < min_scene_trs or onset >= len(isc_z):
            scene_isc_values.append(np.nan)
            continue

        segment = isc_z[onset:min(offset, len(isc_z))]
        if len(segment) == 0:
            scene_isc_values.append(np.nan)
            continue

        if summary_method == "median":
            scene_isc_values.append(float(np.median(segment)))
        else:
            scene_isc_values.append(float(np.mean(segment)))

    df["scene_isc"] = scene_isc_values
    return df


def correlate_isc_with_features(
    aligned_df: pd.DataFrame,
    feature_columns: List[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Correlate scene-level ISC with scene features.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of align_isc_to_scenes (must have 'scene_isc' column).
    feature_columns : list of str
        Feature columns to correlate with scene ISC.
    method : str
        "spearman" or "pearson".

    Returns
    -------
    pd.DataFrame
        Correlation results with columns:
        feature, r, p_value, n_scenes, method
    """
    valid = aligned_df.dropna(subset=["scene_isc"])
    results = []

    for feat in feature_columns:
        if feat not in valid.columns:
            continue

        feat_vals = valid[feat].to_numpy()
        isc_vals = valid["scene_isc"].to_numpy()

        # Skip if no variance
        if np.std(feat_vals) == 0 or np.std(isc_vals) == 0:
            results.append({
                "feature": feat,
                "r": np.nan,
                "p_value": np.nan,
                "n_scenes": len(valid),
                "method": method,
            })
            continue

        if method == "pearson":
            r, p = pearsonr(feat_vals, isc_vals)
        else:
            r, p = spearmanr(feat_vals, isc_vals)

        results.append({
            "feature": feat,
            "r": float(r),
            "p_value": float(p),
            "n_scenes": len(valid),
            "method": method,
        })

    return pd.DataFrame(results)


# ── Dynamic FC + Scene alignment ─────────────────────────────────────────────

def align_dfc_to_scenes(
    dfc_windows: List[np.ndarray],
    scenes_df: pd.DataFrame,
    window_trs: int,
    step_trs: int,
) -> pd.DataFrame:
    """Align dynamic FC windows to scene segments.

    For each scene, identify which dFC windows fall within that scene and compute
    the mean within-scene FC variability (standard deviation across windows).

    Parameters
    ----------
    dfc_windows : list of np.ndarray
        List of FC matrices from sliding_windows().
    scenes_df : pd.DataFrame
        Scene annotation with onset_tr and offset_tr columns.
    window_trs : int
        Window size used for dFC computation.
    step_trs : int
        Step size used for dFC computation.

    Returns
    -------
    pd.DataFrame
        scenes_df with added columns:
        n_windows, mean_fc_variability, max_fc_variability
    """
    if not dfc_windows:
        return scenes_df.copy()

    df = scenes_df.copy()

    # Compute the center TR of each window
    window_centers = [
        start + window_trs // 2
        for start in range(0, len(dfc_windows) * step_trs, step_trs)
    ][:len(dfc_windows)]

    # Upper-triangle indices for flattening
    n_rois = dfc_windows[0].shape[0]
    triu_idx = np.triu_indices(n_rois, k=1)

    n_windows_list = []
    mean_var_list = []
    max_var_list = []

    for _, row in df.iterrows():
        onset = int(row["onset_tr"])
        offset = int(row["offset_tr"])

        # Find windows whose center falls within this scene
        win_indices = [
            i for i, center in enumerate(window_centers)
            if onset <= center < offset
        ]

        if len(win_indices) < 2:
            n_windows_list.append(len(win_indices))
            mean_var_list.append(np.nan)
            max_var_list.append(np.nan)
            continue

        # Stack the FC matrices for these windows
        scene_mats = np.stack([dfc_windows[i] for i in win_indices])
        # Compute variability (std across windows) for each edge
        edge_std = np.std(scene_mats, axis=0)

        n_windows_list.append(len(win_indices))
        mean_var_list.append(float(np.mean(edge_std[triu_idx])))
        max_var_list.append(float(np.max(edge_std[triu_idx])))

    df["n_dfc_windows"] = n_windows_list
    df["mean_fc_variability"] = mean_var_list
    df["max_fc_variability"] = max_var_list

    return df


# ── Visualization ────────────────────────────────────────────────────────────

def plot_isc_scene_alignment(
    aligned_df: pd.DataFrame,
    feature: str,
    out_file: str,
    title: Optional[str] = None,
) -> None:
    """Plot ISC vs a scene feature as a scatter with regression line.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of align_isc_to_scenes.
    feature : str
        Feature column name to plot on x-axis.
    out_file : str
        Path to save the figure.
    title : str, optional
        Plot title.
    """
    valid = aligned_df.dropna(subset=["scene_isc", feature])
    if len(valid) < 3:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    x = valid[feature].to_numpy()
    y = valid["scene_isc"].to_numpy()

    ax.scatter(x, y, alpha=0.6, edgecolors="white", s=60, c="#2c3e50")

    # Add regression line
    if np.std(x) > 0 and np.std(y) > 0:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

        r, pval = spearmanr(x, y)
        ax.text(
            0.05, 0.95,
            f"Spearman r = {r:.3f}\np = {pval:.4f}\nn = {len(valid)} scenes",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_xlabel(feature.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel("Scene ISC (z-scored)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    else:
        ax.set_title(f"ISC vs {feature.replace('_', ' ').title()}", fontsize=12, fontweight="bold")

    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


def plot_isc_timecourse_with_scenes(
    isc_timecourse: np.ndarray,
    scenes_df: pd.DataFrame,
    feature: str,
    tr_sec: float,
    out_file: str,
    title: Optional[str] = None,
) -> None:
    """Plot ISC timecourse with scene boundaries and feature annotations.

    Parameters
    ----------
    isc_timecourse : np.ndarray
        1D array of ISC values per TR.
    scenes_df : pd.DataFrame
        Scene annotation with onset_tr, offset_tr, and the given feature column.
    feature : str
        Feature column for color-coding scene segments.
    tr_sec : float
        TR in seconds for x-axis labeling.
    out_file : str
        Path to save the figure.
    title : str, optional
        Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={"hspace": 0.05})

    t = np.arange(len(isc_timecourse)) * tr_sec

    # Top: ISC timecourse
    ax1.plot(t, isc_timecourse, color="#2c3e50", linewidth=1, alpha=0.8)
    ax1.fill_between(t, isc_timecourse, alpha=0.15, color="#3498db")
    ax1.set_ylabel("ISC", fontsize=11)
    if title:
        ax1.set_title(title, fontsize=12, fontweight="bold")

    # Add scene boundaries as vertical lines
    if "onset_tr" in scenes_df.columns:
        for _, row in scenes_df.iterrows():
            onset_sec = row["onset_tr"] * tr_sec
            ax1.axvline(onset_sec, color="gray", alpha=0.3, linewidth=0.5)

    # Bottom: feature annotation
    if feature in scenes_df.columns and "onset_tr" in scenes_df.columns:
        cmap = plt.cm.RdYlGn
        for _, row in scenes_df.iterrows():
            onset_sec = row["onset_tr"] * tr_sec
            offset_sec = row["offset_tr"] * tr_sec
            val = row[feature]
            if np.isnan(val):
                color = "lightgray"
            else:
                # Normalize to [0, 1] range for colormap
                norm_val = (val + 1) / 2 if feature == "emotional_valence" else val
                color = cmap(np.clip(norm_val, 0, 1))
            ax2.axvspan(onset_sec, offset_sec, color=color, alpha=0.7)

    ax2.set_xlabel("Time (seconds)", fontsize=11)
    ax2.set_ylabel(feature.replace("_", " ").title(), fontsize=9)
    ax2.set_yticks([])

    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()


# ── Template generation ──────────────────────────────────────────────────────

def create_annotation_template(
    output_path: str,
    features: Optional[List[str]] = None,
    n_example_rows: int = 5,
) -> str:
    """Create a blank scene annotation CSV template.

    Parameters
    ----------
    output_path : str
        Where to save the template CSV.
    features : list of str, optional
        Feature columns to include. Defaults to the three core features.
    n_example_rows : int
        Number of example rows with placeholder values.

    Returns
    -------
    str
        Path to the created template.
    """
    if features is None:
        features = ["emotional_valence", "social_cognition", "narrative_transition"]

    # Create example data
    rows = []
    for i in range(n_example_rows):
        row = {
            "onset_sec": float(i * 30),
            "offset_sec": float((i + 1) * 30),
        }
        for feat in features:
            if feat == "emotional_valence":
                row[feat] = 0.0  # placeholder
            elif feat in ("social_cognition", "narrative_transition"):
                row[feat] = 0  # placeholder
            else:
                row[feat] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
