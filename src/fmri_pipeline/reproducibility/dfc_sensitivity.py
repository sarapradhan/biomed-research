"""Dynamic FC window-size sensitivity (Phase 2 of the JEI revision plan).

For each ROI timeseries, runs sliding-window dynamic FC at multiple window
sizes (default 20, 30, 40 TR), summarises FC variability across windows,
and computes the Adjusted Rand Index between k-means state-assignment
label sequences from different window sizes (after projecting them onto
a common per-TR timeline).

The motivating question is whether the qualitative interpretation of
dynamic FC is robust across reasonable window choices. Conservative
interpretation expectation: per-pair ARI should be modestly positive but
not perfect; FC variability should be of comparable magnitude across
window sizes.

Primary outputs (per group):
    reports/reproducibility/dfc_sensitivity_per_window.csv
    reports/reproducibility/dfc_sensitivity_ari.csv
    reports/reproducibility/dfc_sensitivity.json
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.metrics import adjusted_rand_score

# Reuse the project's vetted static FC implementation so we get the same
# Fisher-z transform, NaN handling, and constant-ROI guard everywhere.
from fmri_pipeline.connectivity import static_fc


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SubjectRunROI:
    """ROI timeseries for one subject/run."""

    subject_id: str
    run_id: str
    timeseries: np.ndarray  # shape (n_TRs, n_rois)


@dataclass
class WindowSweepResult:
    """Summary for one window size on one run."""

    window_size: int
    step_size: int
    n_windows: int
    mean_fc_variability: float
    median_fc_variability: float
    k_states: int
    state_labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


@dataclass
class RunSensitivityResult:
    """All window-size sweeps + pairwise ARI for one run."""

    subject_id: str
    run_id: str
    n_TRs: int
    n_rois: int
    window_sizes: List[int]
    sweeps: Dict[int, WindowSweepResult]
    ari_pairs: List[Dict[str, Any]]  # {"w1", "w2", "ari", "n_TRs_compared"}


@dataclass
class GroupSensitivityResult:
    """Aggregate summary across all runs."""

    window_sizes: List[int]
    n_runs: int
    per_window_variability_mean: Dict[int, float]
    per_window_variability_std: Dict[int, float]
    pairwise_ari_mean: Dict[str, float]   # key "W1-W2"
    pairwise_ari_std: Dict[str, float]
    runs: List[RunSensitivityResult] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Window helpers
# --------------------------------------------------------------------------- #
def window_starts(n_TRs: int, window_size: int, step_size: int) -> np.ndarray:
    """Indices of the first TR of each sliding window."""
    if window_size > n_TRs or window_size < 2 or step_size < 1:
        return np.zeros(0, dtype=int)
    return np.arange(0, n_TRs - window_size + 1, step_size, dtype=int)


def window_centers(n_TRs: int, window_size: int, step_size: int) -> np.ndarray:
    """Center TR of each window (rounded to integer, clipped to valid range)."""
    starts = window_starts(n_TRs, window_size, step_size)
    if starts.size == 0:
        return starts
    centers = np.round(starts + (window_size - 1) / 2.0).astype(int)
    return np.clip(centers, 0, n_TRs - 1)


def project_labels_to_TR(
    labels: np.ndarray,
    n_TRs: int,
    window_size: int,
    step_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project per-window labels to per-TR labels.

    For each TR t we assign the label of the window whose center is
    closest to t. TRs outside the [first center, last center] range are
    masked out (returned in ``valid_mask``).

    Returns
    -------
    per_tr_labels : np.ndarray of int, shape (n_TRs,)
        ``-1`` outside the valid range; otherwise the label of the
        nearest-center window.
    valid_mask : np.ndarray of bool, shape (n_TRs,)
    """
    centers = window_centers(n_TRs, window_size, step_size)
    if centers.size == 0:
        return np.full(n_TRs, -1, dtype=int), np.zeros(n_TRs, dtype=bool)
    if centers.size != labels.size:
        raise ValueError(
            f"labels length {labels.size} does not match number of windows "
            f"{centers.size}."
        )

    per_tr = np.full(n_TRs, -1, dtype=int)
    first, last = int(centers[0]), int(centers[-1])
    valid_mask = np.zeros(n_TRs, dtype=bool)
    for t in range(first, last + 1):
        idx = int(np.argmin(np.abs(centers - t)))
        per_tr[t] = int(labels[idx])
        valid_mask[t] = True
    return per_tr, valid_mask


# --------------------------------------------------------------------------- #
# Per-window FC sweep
# --------------------------------------------------------------------------- #
def _upper_triangle(mat: np.ndarray, k: int = 1) -> np.ndarray:
    idx = np.triu_indices_from(mat, k=k)
    return mat[idx]


def windowed_fc_matrices(
    ts: np.ndarray, window_size: int, step_size: int
) -> List[np.ndarray]:
    """Compute per-window static FC matrices using ``connectivity.static_fc``."""
    starts = window_starts(ts.shape[0], window_size, step_size)
    return [static_fc(ts[s : s + window_size]) for s in starts]


def fc_variability_across_windows(mats: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Mean and median std of upper-triangle FC values across windows.

    Returns ``(0.0, 0.0)`` if fewer than two windows are available.
    """
    if len(mats) < 2:
        return 0.0, 0.0
    stack = np.stack([_upper_triangle(m) for m in mats], axis=0)
    sd = np.std(stack, axis=0, ddof=1)
    return float(np.mean(sd)), float(np.median(sd))


def cluster_window_states(
    mats: Sequence[np.ndarray], k: int, seed: int
) -> np.ndarray:
    """k-means cluster window-FC vectors. Returns label per window.

    Uses scipy's ``kmeans2`` with the project-wide seed for determinism.
    Empty cluster fallback: if k > len(mats) or kmeans2 returns degenerate
    output, returns a 0-vector of length len(mats).
    """
    if not mats:
        return np.zeros(0, dtype=int)
    if k <= 1 or len(mats) < k:
        return np.zeros(len(mats), dtype=int)
    triu = np.triu_indices_from(mats[0], k=1)
    x = np.stack([m[triu] for m in mats], axis=0)
    _, labels = kmeans2(x, k, minit="points", seed=int(seed))
    return labels.astype(int)


def sweep_window_size(
    ts: np.ndarray,
    window_size: int,
    step_size: int,
    k_states: int,
    seed: int,
) -> WindowSweepResult:
    """Run dFC + clustering for one window size."""
    mats = windowed_fc_matrices(ts, window_size, step_size)
    mean_var, med_var = fc_variability_across_windows(mats)
    labels = cluster_window_states(mats, k_states, seed=seed)
    return WindowSweepResult(
        window_size=int(window_size),
        step_size=int(step_size),
        n_windows=len(mats),
        mean_fc_variability=mean_var,
        median_fc_variability=med_var,
        k_states=int(k_states),
        state_labels=labels,
    )


# --------------------------------------------------------------------------- #
# Pairwise ARI
# --------------------------------------------------------------------------- #
def pairwise_ari(
    sweeps: Dict[int, WindowSweepResult], n_TRs: int
) -> List[Dict[str, Any]]:
    """ARI between every pair of window sizes after per-TR projection."""
    out: List[Dict[str, Any]] = []
    sizes = sorted(sweeps.keys())
    for w1, w2 in combinations(sizes, 2):
        s1 = sweeps[w1]
        s2 = sweeps[w2]
        # Skip pairs where either sweep produced no clustering (e.g. too few windows).
        if s1.state_labels.size == 0 or s2.state_labels.size == 0:
            out.append({"w1": w1, "w2": w2, "ari": float("nan"), "n_TRs_compared": 0})
            continue
        l1, m1 = project_labels_to_TR(s1.state_labels, n_TRs, w1, s1.step_size)
        l2, m2 = project_labels_to_TR(s2.state_labels, n_TRs, w2, s2.step_size)
        common = m1 & m2
        if not common.any():
            out.append({"w1": w1, "w2": w2, "ari": float("nan"), "n_TRs_compared": 0})
            continue
        ari = float(adjusted_rand_score(l1[common], l2[common]))
        out.append(
            {
                "w1": int(w1),
                "w2": int(w2),
                "ari": ari,
                "n_TRs_compared": int(common.sum()),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyse_run(
    run: SubjectRunROI,
    window_sizes: Sequence[int],
    step_size: int,
    k_states: int,
    seed: int,
) -> RunSensitivityResult:
    """Sweep all window sizes for one run and compute pairwise ARI."""
    sweeps: Dict[int, WindowSweepResult] = {}
    for w in window_sizes:
        sweeps[int(w)] = sweep_window_size(
            run.timeseries,
            window_size=int(w),
            step_size=int(step_size),
            k_states=int(k_states),
            seed=int(seed),
        )
    ari = pairwise_ari(sweeps, n_TRs=run.timeseries.shape[0])
    return RunSensitivityResult(
        subject_id=run.subject_id,
        run_id=run.run_id,
        n_TRs=int(run.timeseries.shape[0]),
        n_rois=int(run.timeseries.shape[1]),
        window_sizes=[int(w) for w in window_sizes],
        sweeps=sweeps,
        ari_pairs=ari,
    )


# --------------------------------------------------------------------------- #
# Group aggregation
# --------------------------------------------------------------------------- #
def aggregate_group(
    runs: Sequence[RunSensitivityResult], window_sizes: Sequence[int]
) -> GroupSensitivityResult:
    """Mean/SD of FC variability per window size and ARI per pair, across runs."""
    sizes = [int(w) for w in window_sizes]
    var_mean: Dict[int, float] = {}
    var_std: Dict[int, float] = {}
    for w in sizes:
        vals = np.array(
            [r.sweeps[w].mean_fc_variability for r in runs if w in r.sweeps],
            dtype=float,
        )
        if vals.size:
            var_mean[w] = float(vals.mean())
            var_std[w] = float(vals.std(ddof=1) if vals.size > 1 else 0.0)
        else:
            var_mean[w] = float("nan")
            var_std[w] = float("nan")

    ari_mean: Dict[str, float] = {}
    ari_std: Dict[str, float] = {}
    for w1, w2 in combinations(sizes, 2):
        key = f"{w1}-{w2}"
        vals: List[float] = []
        for r in runs:
            for rec in r.ari_pairs:
                if rec["w1"] == w1 and rec["w2"] == w2 and not np.isnan(rec["ari"]):
                    vals.append(rec["ari"])
        arr = np.asarray(vals, dtype=float)
        if arr.size:
            ari_mean[key] = float(arr.mean())
            ari_std[key] = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
        else:
            ari_mean[key] = float("nan")
            ari_std[key] = float("nan")

    return GroupSensitivityResult(
        window_sizes=sizes,
        n_runs=len(runs),
        per_window_variability_mean=var_mean,
        per_window_variability_std=var_std,
        pairwise_ari_mean=ari_mean,
        pairwise_ari_std=ari_std,
        runs=list(runs),
    )


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_run_timeseries(input_dir: Path) -> List[SubjectRunROI]:
    """Load ROI timeseries saved as ``sub-<id>_run-<id>_roi.npy`` (T, n_rois)."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"ROI input directory does not exist: {input_dir}")
    out: List[SubjectRunROI] = []
    for path in sorted(input_dir.glob("sub-*_run-*_roi.npy")):
        parts = path.stem.split("_")
        sub_entity = next(p for p in parts if p.startswith("sub-"))
        run_entity = next(p for p in parts if p.startswith("run-"))
        ses_entity = next((p for p in parts if p.startswith("ses-")), None)
        subject_id = (
            f"{sub_entity.replace('sub-', '')}__{ses_entity}"
            if ses_entity
            else sub_entity.replace("sub-", "")
        )
        ts = np.load(path)
        if ts.ndim != 2:
            raise ValueError(
                f"Expected 2D ROI timeseries (T, n_rois) at {path}; "
                f"got shape {ts.shape}."
            )
        out.append(
            SubjectRunROI(
                subject_id=subject_id,
                run_id=run_entity.replace("run-", ""),
                timeseries=ts.astype(float),
            )
        )
    return out


def write_outputs(
    group: GroupSensitivityResult,
    output_dir: Path,
    per_window_csv: str = "dfc_sensitivity_per_window.csv",
    ari_csv: str = "dfc_sensitivity_ari.csv",
    json_filename: str = "dfc_sensitivity.json",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_path = output_dir / per_window_csv
    ari_path = output_dir / ari_csv
    json_path = output_dir / json_filename

    with per_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "subject_id",
                "run_id",
                "window_size",
                "step_size",
                "n_windows",
                "mean_fc_variability",
                "median_fc_variability",
                "k_states",
            ]
        )
        for r in group.runs:
            for size in group.window_sizes:
                s = r.sweeps[size]
                w.writerow(
                    [
                        r.subject_id,
                        r.run_id,
                        s.window_size,
                        s.step_size,
                        s.n_windows,
                        s.mean_fc_variability,
                        s.median_fc_variability,
                        s.k_states,
                    ]
                )
        # Group means appended at the end with a sentinel "GROUP" subject.
        for size in group.window_sizes:
            w.writerow(
                [
                    "GROUP",
                    "MEAN",
                    size,
                    "",
                    "",
                    group.per_window_variability_mean.get(size, float("nan")),
                    "",
                    "",
                ]
            )

    with ari_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "run_id", "w1", "w2", "ari", "n_TRs_compared"])
        for r in group.runs:
            for rec in r.ari_pairs:
                w.writerow(
                    [r.subject_id, r.run_id, rec["w1"], rec["w2"], rec["ari"], rec["n_TRs_compared"]]
                )
        for key, mean in group.pairwise_ari_mean.items():
            w1_str, w2_str = key.split("-")
            w.writerow(["GROUP", "MEAN", w1_str, w2_str, mean, ""])

    # JSON snapshot — serialise dataclasses, converting numpy arrays to lists.
    def _serialise(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(repr(o))

    snapshot = {
        "window_sizes": group.window_sizes,
        "n_runs": group.n_runs,
        "per_window_variability_mean": group.per_window_variability_mean,
        "per_window_variability_std": group.per_window_variability_std,
        "pairwise_ari_mean": group.pairwise_ari_mean,
        "pairwise_ari_std": group.pairwise_ari_std,
        "runs": [
            {
                "subject_id": r.subject_id,
                "run_id": r.run_id,
                "n_TRs": r.n_TRs,
                "n_rois": r.n_rois,
                "window_sizes": r.window_sizes,
                "sweeps": {
                    int(w): {
                        k: v
                        for k, v in asdict(s).items()
                        if k != "state_labels"
                    }
                    for w, s in r.sweeps.items()
                },
                "ari_pairs": r.ari_pairs,
            }
            for r in group.runs
        ],
    }
    json_path.write_text(json.dumps(snapshot, indent=2, default=_serialise))

    return {"per_window_csv": per_path, "ari_csv": ari_path, "json": json_path}


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run(config: Dict[str, Any] | str | Path) -> GroupSensitivityResult:
    """Top-level entry point.

    Expected config keys (extension to ``config/reproducibility.yaml``)::

        paths.dfc_input_dir       # ROI timeseries dir (sub-*_run-*_roi.npy)
        paths.output_root
        dfc.window_sizes          (default [20, 30, 40])
        dfc.step_size             (default 1)
        dfc.k_states              (default 4)
        dfc.output_per_window_csv (default dfc_sensitivity_per_window.csv)
        dfc.output_ari_csv        (default dfc_sensitivity_ari.csv)
        dfc.output_json           (default dfc_sensitivity.json)
        project.random_seed       (default 42)
    """
    import yaml

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    cfg = config or {}
    dfc_cfg = cfg.get("dfc", {}) if isinstance(cfg, dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    input_dir = Path(paths["dfc_input_dir"])
    output_dir = Path(paths["output_root"])
    window_sizes = list(dfc_cfg.get("window_sizes", [20, 30, 40]))
    step_size = int(dfc_cfg.get("step_size", 1))
    k_states = int(dfc_cfg.get("k_states", 4))
    seed = int(cfg.get("project", {}).get("random_seed", 42))

    runs = load_run_timeseries(input_dir)
    if not runs:
        raise ValueError(f"No ROI timeseries found in {input_dir}")

    per_run_results = [
        analyse_run(r, window_sizes=window_sizes, step_size=step_size, k_states=k_states, seed=seed)
        for r in runs
    ]
    group = aggregate_group(per_run_results, window_sizes)
    write_outputs(
        group,
        output_dir,
        per_window_csv=dfc_cfg.get("output_per_window_csv", "dfc_sensitivity_per_window.csv"),
        ari_csv=dfc_cfg.get("output_ari_csv", "dfc_sensitivity_ari.csv"),
        json_filename=dfc_cfg.get("output_json", "dfc_sensitivity.json"),
    )
    return group
