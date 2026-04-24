"""Static FC reproducibility.

Computes run-wise static functional connectivity matrices per subject,
extracts the upper triangle, and tests whether within-subject FC
similarity across runs exceeds between-subject similarity across
matched runs.

Primary output:  reports/reproducibility/fc_within_vs_between.csv
Figure panel:    violin/boxplot of within vs between similarity.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SubjectRunFC:
    """Static FC matrix for a single subject/run."""

    subject_id: str
    run_id: str
    fc_matrix: np.ndarray  # shape (n_rois, n_rois), Fisher-z or raw r

    def upper_triangle(self) -> np.ndarray:
        """Return vectorized upper triangle (excluding diagonal)."""
        idx = np.triu_indices_from(self.fc_matrix, k=1)
        return self.fc_matrix[idx]


@dataclass
class ReproducibilityResult:
    """Summary of the within vs between analysis."""

    within_mean: float
    within_std: float
    between_mean: float
    between_std: float
    gap_mean: float
    gap_ci_low: float
    gap_ci_high: float
    n_subjects: int
    n_within_pairs: int
    n_between_pairs: int
    test_statistic: float
    p_value: float
    effect_size: float  # Cohen's d


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fisher_z(r: Sequence[float] | np.ndarray) -> np.ndarray:
    r_arr = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    return np.arctanh(r_arr)


def _group_by_subject(run_fcs: Sequence[SubjectRunFC]) -> Dict[str, List[SubjectRunFC]]:
    grouped: Dict[str, List[SubjectRunFC]] = {}
    for rf in run_fcs:
        grouped.setdefault(rf.subject_id, []).append(rf)
    return grouped


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = (
        (len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)
    ) / (len(a) + len(b) - 2)
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


# --------------------------------------------------------------------------- #
# Core API
# --------------------------------------------------------------------------- #
def compute_within_subject_similarity(
    run_fcs: Sequence[SubjectRunFC],
) -> Dict[str, List[float]]:
    """For each subject with >=2 runs, return Pearson r across all run-pair FC vectors."""
    grouped = _group_by_subject(run_fcs)
    out: Dict[str, List[float]] = {}
    for sid, runs in grouped.items():
        if len(runs) < 2:
            continue
        vecs = [r.upper_triangle() for r in runs]
        out[sid] = [float(np.corrcoef(a, b)[0, 1]) for a, b in combinations(vecs, 2)]
    return out


def compute_between_subject_similarity(
    run_fcs: Sequence[SubjectRunFC],
    match_run_index: bool = True,
) -> List[float]:
    """Pearson r across FC vectors between subject pairs.

    Parameters
    ----------
    match_run_index
        If True, use only same-run-id pairs (reduces degrees of freedom
        asymmetry vs within-subject similarity). Otherwise, average
        across all cross-subject run pairs.
    """
    grouped = _group_by_subject(run_fcs)
    subjects = sorted(grouped.keys())
    sims: List[float] = []

    if match_run_index:
        by_sub_run = {
            s: {r.run_id: r.upper_triangle() for r in grouped[s]} for s in subjects
        }
        for sa, sb in combinations(subjects, 2):
            shared_runs = sorted(set(by_sub_run[sa]) & set(by_sub_run[sb]))
            for run in shared_runs:
                sims.append(
                    float(np.corrcoef(by_sub_run[sa][run], by_sub_run[sb][run])[0, 1])
                )
    else:
        by_sub = {s: [rf.upper_triangle() for rf in grouped[s]] for s in subjects}
        for sa, sb in combinations(subjects, 2):
            for va in by_sub[sa]:
                for vb in by_sub[sb]:
                    sims.append(float(np.corrcoef(va, vb)[0, 1]))
    return sims


def summarize(
    within: Dict[str, List[float]],
    between: List[float],
    n_bootstrap: int = 1000,
    random_state: int | None = 42,
) -> ReproducibilityResult:
    """Summarize within vs between comparison: test, effect size, bootstrap CI on gap."""
    within_vals = [v for vs in within.values() for v in vs]
    if len(within_vals) == 0 or len(between) == 0:
        raise ValueError(
            "summarize() requires at least one within-subject and one "
            "between-subject similarity value"
        )

    w_arr = np.asarray(within_vals, dtype=float)
    b_arr = np.asarray(between, dtype=float)

    # Use Fisher-z scale for the statistical test; report descriptive
    # stats in raw r space for interpretability.
    wz = _fisher_z(w_arr)
    bz = _fisher_z(b_arr)

    if len(w_arr) > 1 and len(b_arr) > 1:
        stat, pval = stats.mannwhitneyu(wz, bz, alternative="greater")
    else:
        stat, pval = float("nan"), float("nan")

    within_mean = float(np.mean(w_arr))
    within_std = float(np.std(w_arr, ddof=1) if len(w_arr) > 1 else 0.0)
    between_mean = float(np.mean(b_arr))
    between_std = float(np.std(b_arr, ddof=1) if len(b_arr) > 1 else 0.0)
    d = _cohens_d(wz, bz)

    # Bootstrap the within-between gap on the raw r scale so the CI is
    # on the reported scale.
    rng = np.random.default_rng(random_state)
    gaps = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        w_boot = rng.choice(w_arr, size=len(w_arr), replace=True)
        b_boot = rng.choice(b_arr, size=len(b_arr), replace=True)
        gaps[i] = w_boot.mean() - b_boot.mean()
    ci_low, ci_high = np.percentile(gaps, [2.5, 97.5])

    return ReproducibilityResult(
        within_mean=within_mean,
        within_std=within_std,
        between_mean=between_mean,
        between_std=between_std,
        gap_mean=within_mean - between_mean,
        gap_ci_low=float(ci_low),
        gap_ci_high=float(ci_high),
        n_subjects=len(within),
        n_within_pairs=len(w_arr),
        n_between_pairs=len(b_arr),
        test_statistic=float(stat) if not np.isnan(stat) else float("nan"),
        p_value=float(pval) if not np.isnan(pval) else float("nan"),
        effect_size=float(d),
    )


def run(config: Dict[str, Any] | str | Path) -> ReproducibilityResult:
    """Top-level entry point.

    Expected config keys (see ``config/reproducibility.yaml``):
        paths.fc_input_dir, paths.output_root
        fc.match_between_by_run_index, fc.n_bootstrap
    """
    import yaml  # local import so the module is importable without PyYAML

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    fc_cfg = config.get("fc", {}) if config else {}
    paths = config.get("paths", {}) if config else {}

    input_dir = Path(paths["fc_input_dir"])
    output_dir = Path(paths["output_root"])

    run_fcs = load_run_fcs(input_dir)
    if len(run_fcs) < 2:
        raise ValueError(
            f"Need at least 2 run-level FC matrices; found {len(run_fcs)} in {input_dir}"
        )

    within = compute_within_subject_similarity(run_fcs)
    between = compute_between_subject_similarity(
        run_fcs,
        match_run_index=fc_cfg.get("match_between_by_run_index", True),
    )
    result = summarize(
        within,
        between,
        n_bootstrap=int(fc_cfg.get("n_bootstrap", 1000)),
        random_state=int(config.get("project", {}).get("random_seed", 42)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(result, output_dir, output_csv=fc_cfg.get("output_csv"))
    return result


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_run_fcs(input_dir: Path) -> List[SubjectRunFC]:
    """Load per-subject, per-run FC matrices.

    Expected layout::

        {input_dir}/sub-<subject>_run-<run>_fc.npy

    Additional entities (e.g., ``_ses-<N>``) are preserved in ``subject_id``
    by using ``sub-<subject>[_ses-<N>]`` as the composite id.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"FC input directory does not exist: {input_dir}")
    fcs: List[SubjectRunFC] = []
    for path in sorted(input_dir.glob("sub-*_run-*_fc.npy")):
        parts = path.stem.split("_")
        sub_entity = next(p for p in parts if p.startswith("sub-"))
        run_entity = next(p for p in parts if p.startswith("run-"))
        ses_entity = next((p for p in parts if p.startswith("ses-")), None)
        subject_id = (
            f"{sub_entity}__{ses_entity}" if ses_entity else sub_entity
        ).replace("sub-", "").replace("ses-", "ses-")
        run_id = run_entity.replace("run-", "")
        fcs.append(
            SubjectRunFC(
                subject_id=subject_id,
                run_id=run_id,
                fc_matrix=np.load(path),
            )
        )
    return fcs


def write_summary(
    result: ReproducibilityResult,
    output_dir: Path,
    output_csv: str | None = None,
) -> Path:
    """Write summary CSV and companion JSON manifest. Returns the CSV path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (output_csv or "fc_within_vs_between.csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in asdict(result).items():
            w.writerow([k, v])
    (output_dir / (csv_path.stem + ".json")).write_text(
        json.dumps(asdict(result), indent=2)
    )
    return csv_path
