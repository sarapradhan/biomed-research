from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from .preprocessing import compute_dvars, compute_tsnr


def plot_fd_trace(fd: np.ndarray, keep_mask: np.ndarray, out_file: str) -> None:
    """Plot FD time series with scrub threshold highlighting."""
    plt.figure(figsize=(10, 3))
    x = np.arange(fd.size)
    plt.plot(x, fd, label="FD (mm)", color="black", linewidth=1.0)
    plt.scatter(x[~keep_mask], fd[~keep_mask], color="red", s=8, label="Scrubbed")
    plt.axhline(0.5, color="orange", linestyle="--", linewidth=1, label="FD threshold")
    plt.xlabel("Frame")
    plt.ylabel("FD (mm)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_scrub_mask(keep_mask: np.ndarray, out_file: str) -> None:
    """Plot binary scrub mask."""
    plt.figure(figsize=(10, 1.8))
    plt.imshow(keep_mask[np.newaxis, :], aspect="auto", cmap="Greys", interpolation="nearest")
    plt.yticks([])
    plt.xlabel("Frame")
    plt.title("Scrub Mask (white=kept, black=scrubbed)")
    plt.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=150)
    plt.close()


def summarize_qc_row(
    subject: str,
    dataset: str,
    site: str,
    run_key: str,
    diagnosis: str,
    fd: np.ndarray,
    keep_mask: np.ndarray,
    motion_metrics: Dict[str, float],
    cleaned_img_file: str,
    mask_file: str,
) -> Dict:
    """Build QC summary row for a single run."""
    img = nib.load(cleaned_img_file)
    arr = img.get_fdata(dtype=np.float32)
    mask = nib.load(mask_file).get_fdata()
    return {
        "subject": subject,
        "dataset": dataset,
        "site": site,
        "run_key": run_key,
        "diagnosis": diagnosis,
        "mean_fd": float(np.mean(fd)),
        "median_fd": float(np.median(fd)),
        "percent_scrubbed": motion_metrics["percent_scrubbed"],
        "max_translation_mm": motion_metrics["max_translation_mm"],
        "max_rotation_deg": motion_metrics["max_rotation_deg"],
        "dvars": compute_dvars(arr),
        "tsnr": compute_tsnr(arr, mask),
        "exclude": bool(motion_metrics["exclude"]),
    }


def save_qc_summary(qc_rows: list, output_root: str) -> pd.DataFrame:
    """Save QC summary table as CSV and parquet."""
    df = pd.DataFrame(qc_rows)
    qc_dir = Path(output_root) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(qc_dir / "qc_summary.csv", index=False)
    try:
        df.to_parquet(qc_dir / "qc_summary.parquet", index=False)
    except Exception:
        pass
    return df


def plot_qc_distributions(qc_df: pd.DataFrame, output_root: str) -> None:
    """Plot motion distributions by diagnosis to inspect imbalance."""
    if qc_df.empty or "diagnosis" not in qc_df.columns:
        return

    out_dir = Path(output_root) / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in ["mean_fd", "percent_scrubbed", "dvars", "tsnr"]:
        if col not in qc_df.columns:
            continue
        plt.figure(figsize=(7, 4))
        groups = [g[col].dropna().to_numpy() for _, g in qc_df.groupby("diagnosis")]
        labels = [str(k) for k, _ in qc_df.groupby("diagnosis")]
        if not groups:
            plt.close()
            continue
        plt.boxplot(groups, labels=labels)
        plt.title(f"{col} by Diagnosis")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_by_diagnosis.png", dpi=150)
        plt.close()


def aggregate_roi_qc_reports(roi_df: "pd.DataFrame", output_root: str) -> "pd.DataFrame":
    """Aggregate per-run roi_qc_report.csv files into a global summary.

    Reads ``roi_qc_report.csv`` from each run's ROI output directory,
    annotates rows with subject / run_key metadata, and concatenates them
    into ``qc/roi_qc_summary.csv``.  A high-level per-run count table is
    also written to ``qc/roi_qc_run_summary.csv``.

    Called at the end of the pipeline after all ROI extraction is complete.

    Parameters
    ----------
    roi_df : pd.DataFrame
        Output of ``run_roi_step`` — must contain columns ``subject``,
        ``run_key``, and ``roi_npy`` (used to locate the report sibling).
    output_root : str
        Pipeline output root directory (same as ``cfg["paths"]["output_root"]``).

    Returns
    -------
    pd.DataFrame
        Global per-ROI QC table (all runs concatenated).
    """
    records = []
    run_summary_rows = []

    for _, row in roi_df.iterrows():
        if not isinstance(row.get("roi_npy"), str) or not row["roi_npy"]:
            continue
        report_path = Path(row["roi_npy"]).parent / "roi_qc_report.csv"
        if not report_path.exists():
            continue
        df = pd.read_csv(report_path)
        df.insert(0, "subject", row["subject"])
        df.insert(1, "run_key", row["run_key"])
        records.append(df)

        n_zero = int(df["is_zero_var"].sum())
        n_low  = int(df["is_low_var"].sum())
        zero_idx = df.loc[df["is_zero_var"], "roi_idx"].tolist()
        run_summary_rows.append({
            "subject": row["subject"],
            "run_key": row["run_key"],
            "n_zero_var_rois": n_zero,
            "n_low_var_rois": n_low,
            "zero_var_roi_indices": str(zero_idx),
        })

    qc_dir = Path(output_root) / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if records:
        global_df = pd.concat(records, ignore_index=True)
        global_df.to_csv(qc_dir / "roi_qc_summary.csv", index=False)
    else:
        global_df = pd.DataFrame()

    run_sum_df = pd.DataFrame(run_summary_rows)
    run_sum_df.to_csv(qc_dir / "roi_qc_run_summary.csv", index=False)

    return global_df

