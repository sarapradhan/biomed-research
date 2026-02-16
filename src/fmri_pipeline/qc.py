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
