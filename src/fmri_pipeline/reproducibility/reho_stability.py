"""ReHo run-to-run stability.

Computes ROI-level ReHo summaries per run, correlates run-wise ReHo
vectors within subject and between subjects, and tests that the within
> between gap is significant. Implements Step 1.2 of the JEI revision
plan (``_archive/docs/revision/JEI_Revision_Implementation_Plan.md``).

The statistical procedure mirrors ``fc_reproducibility``: Fisher-z
transform before parametric/non-parametric tests, one-sided
Mann-Whitney for the within > between contrast, Cohen's d on
Fisher-z values, and a percentile bootstrap CI on the within-between
gap reported in raw r space.

Primary outputs:
    reports/reproducibility/reho_similarity_matrix.npy  (n_runs x n_runs)
    reports/reproducibility/reho_summary.csv
    reports/reproducibility/reho_summary.json
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SubjectRunReHo:
    """ROI-level ReHo vector for a single subject/run."""

    subject_id: str
    run_id: str
    reho_roi_vector: np.ndarray  # shape (n_rois,)


@dataclass
class ReHoStabilityResult:
    """Summary of within vs between ReHo similarity."""

    within_mean: float
    within_std: float
    between_mean: float
    between_std: float
    gap_mean: float
    gap_ci_low: float
    gap_ci_high: float
    n_subjects: int
    n_runs: int
    n_within_pairs: int
    n_between_pairs: int
    test_statistic: float
    p_value: float
    effect_size: float            # Cohen's d on Fisher-z values
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    run_index: List[Tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers (kept consistent with fc_reproducibility)
# --------------------------------------------------------------------------- #
def _fisher_z(r: Sequence[float] | np.ndarray) -> np.ndarray:
    r_arr = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    return np.arctanh(r_arr)


def _group_by_subject(rehos: Sequence[SubjectRunReHo]) -> Dict[str, List[SubjectRunReHo]]:
    grouped: Dict[str, List[SubjectRunReHo]] = {}
    for rh in rehos:
        grouped.setdefault(rh.subject_id, []).append(rh)
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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns NaN (not a warning) for zero-variance inputs."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Core API
# --------------------------------------------------------------------------- #
def compute_similarity_matrix(
    run_rehos: Sequence[SubjectRunReHo],
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Pairwise Pearson r between every pair of run-level ReHo vectors.

    Returns
    -------
    matrix : np.ndarray, shape (n_runs, n_runs)
        Symmetric matrix with 1.0 on the diagonal.
    index : list of (subject_id, run_id)
        Row/column labels in the same order as ``matrix``.
    """
    n = len(run_rehos)
    mat = np.eye(n, dtype=float)
    index = [(rh.subject_id, rh.run_id) for rh in run_rehos]
    for i in range(n):
        vi = np.asarray(run_rehos[i].reho_roi_vector, dtype=float)
        for j in range(i + 1, n):
            vj = np.asarray(run_rehos[j].reho_roi_vector, dtype=float)
            r = _safe_corr(vi, vj)
            mat[i, j] = r
            mat[j, i] = r
    return mat, index


def compute_within_subject_similarity(
    run_rehos: Sequence[SubjectRunReHo],
) -> Dict[str, List[float]]:
    """For each subject with >=2 runs, return Pearson r across run-pair ReHo vectors."""
    grouped = _group_by_subject(run_rehos)
    out: Dict[str, List[float]] = {}
    for sid, runs in grouped.items():
        if len(runs) < 2:
            continue
        vecs = [np.asarray(r.reho_roi_vector, dtype=float) for r in runs]
        out[sid] = [_safe_corr(a, b) for a, b in combinations(vecs, 2)]
    return out


def compute_between_subject_similarity(
    run_rehos: Sequence[SubjectRunReHo],
    match_run_index: bool = True,
) -> List[float]:
    """Pearson r across ReHo vectors between subject pairs.

    When ``match_run_index`` is True, only same-run-id pairs across subjects
    are used (keeps run-index degrees of freedom symmetric with the
    within-subject distribution). Otherwise, all cross-subject run pairs
    are used.
    """
    grouped = _group_by_subject(run_rehos)
    subjects = sorted(grouped.keys())
    sims: List[float] = []

    if match_run_index:
        by_sub_run = {
            s: {r.run_id: np.asarray(r.reho_roi_vector, dtype=float) for r in grouped[s]}
            for s in subjects
        }
        for sa, sb in combinations(subjects, 2):
            shared_runs = sorted(set(by_sub_run[sa]) & set(by_sub_run[sb]))
            for run in shared_runs:
                sims.append(_safe_corr(by_sub_run[sa][run], by_sub_run[sb][run]))
    else:
        by_sub = {
            s: [np.asarray(rh.reho_roi_vector, dtype=float) for rh in grouped[s]]
            for s in subjects
        }
        for sa, sb in combinations(subjects, 2):
            for va in by_sub[sa]:
                for vb in by_sub[sb]:
                    sims.append(_safe_corr(va, vb))
    return sims


def summarize(
    within: Dict[str, List[float]],
    between: List[float],
    similarity_matrix: np.ndarray | None = None,
    run_index: List[Tuple[str, str]] | None = None,
    n_subjects: int | None = None,
    n_bootstrap: int = 1000,
    random_state: int | None = 42,
) -> ReHoStabilityResult:
    """Summarize within vs between similarity: test, effect size, bootstrap CI."""
    within_vals = [v for vs in within.values() for v in vs if not np.isnan(v)]
    between_vals = [v for v in between if not np.isnan(v)]
    if len(within_vals) == 0 or len(between_vals) == 0:
        raise ValueError(
            "summarize() requires at least one within-subject and one "
            "between-subject similarity value"
        )

    w_arr = np.asarray(within_vals, dtype=float)
    b_arr = np.asarray(between_vals, dtype=float)
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

    # Bootstrap CI on the gap in raw r space.
    rng = np.random.default_rng(random_state)
    gaps = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        w_boot = rng.choice(w_arr, size=len(w_arr), replace=True)
        b_boot = rng.choice(b_arr, size=len(b_arr), replace=True)
        gaps[i] = w_boot.mean() - b_boot.mean()
    ci_low, ci_high = np.percentile(gaps, [2.5, 97.5])

    sim_mat = (
        similarity_matrix
        if similarity_matrix is not None
        else np.zeros((0, 0), dtype=float)
    )
    idx = run_index or []
    n_runs = sim_mat.shape[0] if sim_mat.size else len(idx)
    n_subj = n_subjects if n_subjects is not None else len(within)

    return ReHoStabilityResult(
        within_mean=within_mean,
        within_std=within_std,
        between_mean=between_mean,
        between_std=between_std,
        gap_mean=within_mean - between_mean,
        gap_ci_low=float(ci_low),
        gap_ci_high=float(ci_high),
        n_subjects=int(n_subj),
        n_runs=int(n_runs),
        n_within_pairs=len(w_arr),
        n_between_pairs=len(b_arr),
        test_statistic=float(stat) if not np.isnan(stat) else float("nan"),
        p_value=float(pval) if not np.isnan(pval) else float("nan"),
        effect_size=float(d),
        similarity_matrix=sim_mat,
        run_index=list(idx),
    )


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_run_reho_vectors(input_dir: Path) -> List[SubjectRunReHo]:
    """Load per-subject, per-run ROI ReHo vectors.

    Expected layout::

        {input_dir}/sub-<subject>_run-<run>_reho.npy   (1D, length n_rois)

    Optional ``ses-<N>`` is folded into ``subject_id`` so within-subject
    pairs are computed within the same session by default.
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"ReHo input directory does not exist: {input_dir}")
    out: List[SubjectRunReHo] = []
    for path in sorted(input_dir.glob("sub-*_run-*_reho.npy")):
        parts = path.stem.split("_")
        sub_entity = next(p for p in parts if p.startswith("sub-"))
        run_entity = next(p for p in parts if p.startswith("run-"))
        ses_entity = next((p for p in parts if p.startswith("ses-")), None)
        subject_id = (
            f"{sub_entity.replace('sub-', '')}__{ses_entity}"
            if ses_entity
            else sub_entity.replace("sub-", "")
        )
        run_id = run_entity.replace("run-", "")
        vec = np.load(path)
        if vec.ndim != 1:
            raise ValueError(
                f"Expected 1D ROI ReHo vector, got shape {vec.shape} for {path}"
            )
        out.append(
            SubjectRunReHo(
                subject_id=subject_id,
                run_id=run_id,
                reho_roi_vector=vec.astype(float),
            )
        )
    return out


def write_outputs(
    result: ReHoStabilityResult,
    output_dir: Path,
    matrix_filename: str = "reho_similarity_matrix.npy",
    csv_filename: str = "reho_summary.csv",
) -> Dict[str, Path]:
    """Write similarity matrix .npy + companion CSV/JSON summary.

    Returns a dict of paths keyed by ``"matrix"``, ``"csv"``, ``"json"``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / matrix_filename
    np.save(matrix_path, result.similarity_matrix)

    # Run-index sidecar (so the heatmap can be labelled later).
    index_path = output_dir / (Path(matrix_filename).stem + "_index.csv")
    with index_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "run_id"])
        w.writerows(result.run_index)

    # Summary CSV/JSON. Mirror fc_reproducibility format so a downstream
    # reader can ingest both with the same code path.
    csv_path = output_dir / csv_filename
    summary_dict = asdict(result)
    summary_dict.pop("similarity_matrix", None)
    summary_dict.pop("run_index", None)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary_dict.items():
            w.writerow([k, v])
    json_path = output_dir / (csv_path.stem + ".json")
    json_path.write_text(json.dumps(summary_dict, indent=2))

    return {"matrix": matrix_path, "csv": csv_path, "json": json_path, "index": index_path}


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def run(config: Dict[str, Any] | str | Path) -> ReHoStabilityResult:
    """Top-level entry point.

    Expected config keys (see ``config/reproducibility.yaml``)::

        paths.reho_input_dir
        paths.output_root
        reho.match_between_by_run_index   (default True)
        reho.n_bootstrap                  (default 1000)
        reho.output_matrix                (default reho_similarity_matrix.npy)
        reho.output_csv                   (default reho_summary.csv)
        project.random_seed               (default 42)
    """
    import yaml  # local so the module is importable without PyYAML at import time

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    reho_cfg = (config or {}).get("reho", {}) if isinstance(config, dict) else {}
    paths = (config or {}).get("paths", {}) if isinstance(config, dict) else {}

    input_dir = Path(paths["reho_input_dir"])
    output_dir = Path(paths["output_root"])

    rehos = load_run_reho_vectors(input_dir)
    if len(rehos) < 2:
        raise ValueError(
            f"Need at least 2 run-level ReHo vectors; found {len(rehos)} in {input_dir}"
        )

    sim_mat, run_index = compute_similarity_matrix(rehos)
    within = compute_within_subject_similarity(rehos)
    between = compute_between_subject_similarity(
        rehos,
        match_run_index=bool(reho_cfg.get("match_between_by_run_index", True)),
    )
    result = summarize(
        within,
        between,
        similarity_matrix=sim_mat,
        run_index=run_index,
        n_subjects=len({r.subject_id for r in rehos}),
        n_bootstrap=int(reho_cfg.get("n_bootstrap", 1000)),
        random_state=int((config or {}).get("project", {}).get("random_seed", 42)),
    )
    write_outputs(
        result,
        output_dir,
        matrix_filename=reho_cfg.get("output_matrix", "reho_similarity_matrix.npy"),
        csv_filename=reho_cfg.get("output_csv", "reho_summary.csv"),
    )
    return result
