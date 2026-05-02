"""ICA component stability.

Reruns ICA with fixed component count across multiple random seeds and
across leave-one-run-out subsets. Matches components across runs using
the Hungarian algorithm on absolute spatial correlation (ICA components
are sign-ambiguous, so we maximize ``|r|`` rather than ``r``), and
reports how many components are recovered with mean spatial r above a
configurable threshold (default 0.7). Implements Step 1.3 of the JEI
revision plan.

Primary outputs:
    reports/reproducibility/ica_stability_seeds.csv
    reports/reproducibility/ica_stability_lorocv.csv
    reports/reproducibility/ica_stability_<sweep>_per_component.csv
    reports/reproducibility/ica_stability_<sweep>_pairwise.csv
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class ICARun:
    """One ICA decomposition.

    ``components`` is a 2D array with one row per component and one column
    per spatial feature (voxel or ROI). The number of features must be the
    same across all ``ICARun`` instances in a stability analysis; the
    number of components ``K`` must also match (we don't try to align
    decompositions of different rank).
    """

    tag: str
    components: np.ndarray  # shape (K, n_features)


@dataclass
class ICAStabilityResult:
    """Summary of one stability sweep (seeds OR run subsets)."""

    sweep: str                    # "seeds" or "run_subsets"
    k_components: int
    robust_threshold: float
    n_runs: int
    pairwise_runs_considered: int  # number of (a, b) pairs aggregated
    mean_matched_correlation: float
    std_matched_correlation: float
    median_matched_correlation: float
    n_robust_components: int       # K_robust where per_component_mean_r > threshold
    per_component_mean_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    per_component_std_r: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pairwise_records: List[Dict[str, Any]] = field(default_factory=list)
    reference_tag: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _zscore_rows(x: np.ndarray) -> np.ndarray:
    """Z-score each row; safe against zero-variance rows (returned as zeros)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    safe = sd.copy()
    safe[safe == 0] = 1.0
    z = (x - mu) / safe
    z[(sd == 0).ravel()] = 0.0
    return z


def _abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise |Pearson r| between rows of ``a`` and rows of ``b``.

    Returns array of shape (a.shape[0], b.shape[0]).
    """
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Spatial dimension mismatch: a has {a.shape[1]} features, "
            f"b has {b.shape[1]}."
        )
    az = _zscore_rows(a)
    bz = _zscore_rows(b)
    n_features = a.shape[1]
    # Pearson r on row-z-scored data is the dot product / N.
    r = (az @ bz.T) / float(n_features)
    return np.abs(r)


# --------------------------------------------------------------------------- #
# Core API
# --------------------------------------------------------------------------- #
def match_components_hungarian(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimal one-to-one matching of components in ``a`` to components in ``b``.

    Maximizes total absolute spatial correlation using
    :func:`scipy.optimize.linear_sum_assignment` on a cost matrix of
    ``-|r|``.

    Parameters
    ----------
    a, b : np.ndarray
        Component matrices of shape (K, n_features). Both must have the
        same number of features. ``K`` may differ; the matching size is
        ``min(K_a, K_b)``.

    Returns
    -------
    row_ind : np.ndarray
        Indices into ``a`` for matched components.
    col_ind : np.ndarray
        Indices into ``b`` for matched components.
    matched_r : np.ndarray
        Absolute spatial correlation of each matched pair, in the order
        returned by ``row_ind``/``col_ind``.
    """
    abs_r = _abs_corr_matrix(a, b)
    # linear_sum_assignment minimizes cost; we want max |r|.
    row_ind, col_ind = linear_sum_assignment(-abs_r)
    matched_r = abs_r[row_ind, col_ind]
    return row_ind, col_ind, matched_r


def _per_pair_summary(
    runs: Sequence[ICARun], threshold: float
) -> List[Dict[str, Any]]:
    """For every unordered pair (a, b), match and record summary stats."""
    records: List[Dict[str, Any]] = []
    for a, b in combinations(runs, 2):
        _, _, matched_r = match_components_hungarian(a.components, b.components)
        records.append(
            {
                "tag_a": a.tag,
                "tag_b": b.tag,
                "k_matched": int(matched_r.size),
                "mean_matched_r": float(np.mean(matched_r)),
                "median_matched_r": float(np.median(matched_r)),
                "min_matched_r": float(np.min(matched_r)),
                "n_above_threshold": int(np.sum(matched_r > threshold)),
            }
        )
    return records


def _per_component_mean_against_reference(
    reference: ICARun, others: Sequence[ICARun]
) -> Tuple[np.ndarray, np.ndarray]:
    """For each component in ``reference``, mean/std |r| of its matched
    counterpart across ``others``.

    Returns ``(mean_r, std_r)`` of shape ``(K_ref,)``.
    """
    K = reference.components.shape[0]
    matched = np.full((K, len(others)), np.nan, dtype=float)
    for j, other in enumerate(others):
        row_ind, _, matched_r = match_components_hungarian(
            reference.components, other.components
        )
        # ``row_ind`` indexes into the reference; ``matched_r`` is aligned to
        # row_ind. Place each matched value at its reference component slot.
        matched[row_ind, j] = matched_r
    # nanmean/nanstd guard against components that couldn't be matched (e.g.
    # K_other < K_ref); those slots stay NaN and are ignored in aggregation.
    with np.errstate(invalid="ignore"):
        mean_r = np.nanmean(matched, axis=1)
        std_r = np.nanstd(matched, axis=1, ddof=0)
    return mean_r, std_r


def _stability_summary(
    runs: Sequence[ICARun],
    sweep: str,
    robust_threshold: float = 0.7,
) -> ICAStabilityResult:
    """Shared aggregation logic for both seed and run-subset sweeps."""
    if len(runs) < 2:
        raise ValueError(
            f"ICA stability requires >=2 decompositions for sweep '{sweep}'; "
            f"got {len(runs)}."
        )
    K_set = {r.components.shape[0] for r in runs}
    if len(K_set) > 1:
        raise ValueError(
            f"All ICARun decompositions must have the same K; got {sorted(K_set)}."
        )
    F_set = {r.components.shape[1] for r in runs}
    if len(F_set) > 1:
        raise ValueError(
            f"All ICARun decompositions must have the same n_features; "
            f"got {sorted(F_set)}."
        )

    pairwise = _per_pair_summary(runs, threshold=robust_threshold)
    # Aggregate matched |r| across every pair, every matched component.
    all_matched: List[float] = []
    for a, b in combinations(runs, 2):
        _, _, matched_r = match_components_hungarian(a.components, b.components)
        all_matched.extend(matched_r.tolist())
    matched_arr = np.asarray(all_matched, dtype=float)

    reference = runs[0]
    others = runs[1:]
    per_comp_mean, per_comp_std = _per_component_mean_against_reference(
        reference, others
    )
    n_robust = int(np.nansum(per_comp_mean > robust_threshold))

    return ICAStabilityResult(
        sweep=sweep,
        k_components=int(reference.components.shape[0]),
        robust_threshold=float(robust_threshold),
        n_runs=len(runs),
        pairwise_runs_considered=int(len(pairwise)),
        mean_matched_correlation=float(np.mean(matched_arr)),
        std_matched_correlation=float(np.std(matched_arr, ddof=1) if matched_arr.size > 1 else 0.0),
        median_matched_correlation=float(np.median(matched_arr)),
        n_robust_components=n_robust,
        per_component_mean_r=per_comp_mean,
        per_component_std_r=per_comp_std,
        pairwise_records=pairwise,
        reference_tag=reference.tag,
        details={"n_features": int(reference.components.shape[1])},
    )


def stability_across_seeds(
    runs: Sequence[ICARun], robust_threshold: float = 0.7
) -> ICAStabilityResult:
    """Pairwise Hungarian-matched spatial correlation across all seed pairs."""
    return _stability_summary(runs, sweep="seeds", robust_threshold=robust_threshold)


def stability_across_run_subsets(
    runs: Sequence[ICARun], robust_threshold: float = 0.7
) -> ICAStabilityResult:
    """Analogous stability across leave-one-run-out (or other) subset decompositions."""
    return _stability_summary(
        runs, sweep="run_subsets", robust_threshold=robust_threshold
    )


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_ica_runs(input_dir: Path, glob_pattern: str = "*_ica_components.npy") -> List[ICARun]:
    """Load ICA decompositions saved as 2D ``.npy`` files.

    Filename convention::

        {tag}_ica_components.npy   (e.g., 'seed-1_ica_components.npy')

    The portion of the filename before ``_ica_components.npy`` is used as
    the run tag.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"ICA input directory does not exist: {input_dir}")
    runs: List[ICARun] = []
    for path in sorted(input_dir.glob(glob_pattern)):
        tag = path.name.replace("_ica_components.npy", "")
        comps = np.load(path)
        if comps.ndim != 2:
            raise ValueError(
                f"Expected 2D component matrix (K, n_features) at {path}; "
                f"got shape {comps.shape}."
            )
        runs.append(ICARun(tag=tag, components=comps.astype(float)))
    return runs


def write_stability_outputs(
    result: ICAStabilityResult,
    output_dir: Path,
    summary_csv: str,
) -> Dict[str, Path]:
    """Write the three companion artifacts for a stability sweep.

    Produces:

    * ``{summary_csv}`` — top-level "metric, value" rows (mirrors the
      ``fc_within_vs_between.csv`` shape so a unified scorecard reader can
      pick it up).
    * ``{stem}_per_component.csv`` — per-component mean/std |r|.
    * ``{stem}_pairwise.csv`` — one row per pair of decompositions.
    * ``{stem}.json`` — full structured snapshot, including arrays.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / summary_csv
    stem = csv_path.stem
    per_comp_path = output_dir / f"{stem}_per_component.csv"
    pairwise_path = output_dir / f"{stem}_pairwise.csv"
    json_path = output_dir / f"{stem}.json"

    summary_dict = asdict(result)
    pairwise = summary_dict.pop("pairwise_records", [])
    per_comp_mean = np.asarray(summary_dict.pop("per_component_mean_r", []))
    per_comp_std = np.asarray(summary_dict.pop("per_component_std_r", []))

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary_dict.items():
            w.writerow([k, v])

    with per_comp_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component_index", "mean_matched_r", "std_matched_r", "robust"])
        for k in range(per_comp_mean.size):
            w.writerow(
                [
                    k,
                    float(per_comp_mean[k]),
                    float(per_comp_std[k]) if k < per_comp_std.size else "",
                    bool(per_comp_mean[k] > result.robust_threshold),
                ]
            )

    with pairwise_path.open("w", newline="") as f:
        if pairwise:
            w = csv.DictWriter(f, fieldnames=list(pairwise[0].keys()))
            w.writeheader()
            for row in pairwise:
                w.writerow(row)
        else:
            f.write("")  # empty file is acceptable

    full = dict(summary_dict)
    full["per_component_mean_r"] = per_comp_mean.tolist()
    full["per_component_std_r"] = per_comp_std.tolist()
    full["pairwise_records"] = pairwise
    json_path.write_text(json.dumps(full, indent=2))

    return {
        "csv": csv_path,
        "per_component_csv": per_comp_path,
        "pairwise_csv": pairwise_path,
        "json": json_path,
    }


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run(config: Dict[str, Any] | str | Path) -> Dict[str, ICAStabilityResult]:
    """Top-level entry point.

    Expected config layout (subset of ``config/reproducibility.yaml``)::

        paths.ica_input_dir       # parent dir for both sub-sweeps
        paths.output_root
        ica.stability.robust_threshold      (default 0.7)
        ica.stability.output_seeds_csv      (default ica_stability_seeds.csv)
        ica.stability.output_lorocv_csv     (default ica_stability_lorocv.csv)
        ica.stability.seeds_subdir          (default 'seeds')
        ica.stability.lorocv_subdir         (default 'lorocv')

    Looks for ``{ica_input_dir}/{seeds_subdir}/*_ica_components.npy`` and
    ``{ica_input_dir}/{lorocv_subdir}/*_ica_components.npy``. Either
    sub-sweep may be absent; only the one(s) present will be reported.
    """
    import yaml

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    cfg = config or {}
    stab_cfg = cfg.get("ica", {}).get("stability", {}) if isinstance(cfg, dict) else {}
    paths = cfg.get("paths", {}) if isinstance(cfg, dict) else {}

    ica_root = Path(paths["ica_input_dir"])
    output_dir = Path(paths["output_root"])
    threshold = float(stab_cfg.get("robust_threshold", 0.7))
    seeds_subdir = stab_cfg.get("seeds_subdir", "seeds")
    lorocv_subdir = stab_cfg.get("lorocv_subdir", "lorocv")
    seeds_csv = stab_cfg.get("output_seeds_csv", "ica_stability_seeds.csv")
    lorocv_csv = stab_cfg.get("output_lorocv_csv", "ica_stability_lorocv.csv")

    results: Dict[str, ICAStabilityResult] = {}

    seeds_dir = ica_root / seeds_subdir
    if seeds_dir.exists():
        seed_runs = load_ica_runs(seeds_dir)
        if len(seed_runs) >= 2:
            res = stability_across_seeds(seed_runs, robust_threshold=threshold)
            write_stability_outputs(res, output_dir, summary_csv=seeds_csv)
            results["seeds"] = res

    lorocv_dir = ica_root / lorocv_subdir
    if lorocv_dir.exists():
        sub_runs = load_ica_runs(lorocv_dir)
        if len(sub_runs) >= 2:
            res = stability_across_run_subsets(sub_runs, robust_threshold=threshold)
            write_stability_outputs(res, output_dir, summary_csv=lorocv_csv)
            results["run_subsets"] = res

    if not results:
        raise FileNotFoundError(
            f"No ICA decompositions found under {ica_root}/{{{seeds_subdir},{lorocv_subdir}}}"
        )
    return results
