"""Graph metric stability.

Computes modularity (via Louvain), global efficiency, and mean weighted
clustering coefficient per run at multiple density thresholds. Reports
bootstrap 95% CIs and leave-one-run-out variability. Flags any metric
whose coefficient of variation exceeds 20%.

Primary output:  reports/reproducibility/graph_metrics_bootstrap.csv
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover - guarded by _ensure_networkx
    _HAS_NX = False


METRICS = ("modularity", "global_efficiency", "clustering")


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class SubjectRunGraph:
    subject_id: str
    run_id: str
    adjacency: np.ndarray          # symmetric, zero-diagonal
    density_threshold: float


@dataclass
class GraphMetricSummary:
    metric_name: str
    density_threshold: float
    mean: float
    std: float
    ci_low: float
    ci_high: float
    coefficient_of_variation: float
    n_resamples: int
    method: str = "bootstrap"
    notes: str = ""


@dataclass
class GraphStabilityResult:
    primary_threshold: float
    thresholds_considered: List[float]
    summaries: List[GraphMetricSummary] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ensure_networkx() -> None:
    if not _HAS_NX:  # pragma: no cover
        raise ImportError(
            "networkx >= 3.0 is required for graph stability analysis. "
            "Install with `pip install networkx>=3.0`."
        )


def _symmetrize(m: np.ndarray) -> np.ndarray:
    out = (m + m.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def threshold_adjacency(fc_matrix: np.ndarray, density: float) -> np.ndarray:
    """Return a symmetric zero-diagonal adjacency with approximately ``density`` of edges.

    Keeps the top ``density`` fraction of edges by absolute weight. Edge
    weights below the threshold are set to zero; surviving weights keep
    their original sign.
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1]; got {density}")
    if fc_matrix.ndim != 2 or fc_matrix.shape[0] != fc_matrix.shape[1]:
        raise ValueError("fc_matrix must be square")

    n = fc_matrix.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    weights = np.abs(fc_matrix[triu_idx])
    n_edges = weights.size
    n_keep = max(1, int(round(density * n_edges)))
    if n_keep >= n_edges:
        threshold = 0.0
    else:
        threshold = np.sort(weights)[n_edges - n_keep]

    mask = np.abs(fc_matrix) >= threshold
    adj = np.where(mask, fc_matrix, 0.0)
    return _symmetrize(adj)


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #
def compute_metrics(graph: SubjectRunGraph, seed: int = 42) -> Dict[str, float]:
    """Compute modularity, global efficiency, and weighted clustering."""
    _ensure_networkx()
    adj = np.asarray(graph.adjacency, dtype=float)
    abs_adj = np.abs(adj)
    np.fill_diagonal(abs_adj, 0.0)

    # Weighted graph for modularity & clustering
    G_w = nx.from_numpy_array(abs_adj)
    # Unweighted (binarized) graph for global efficiency path lengths
    binary = (abs_adj > 0).astype(float)
    G_b = nx.from_numpy_array(binary)

    out: Dict[str, float] = {}
    try:
        communities = nx.community.louvain_communities(G_w, seed=seed)
        out["modularity"] = float(nx.community.modularity(G_w, communities))
    except Exception:  # noqa: BLE001
        out["modularity"] = float("nan")

    try:
        out["global_efficiency"] = float(nx.global_efficiency(G_b))
    except Exception:  # noqa: BLE001
        out["global_efficiency"] = float("nan")

    try:
        out["clustering"] = float(nx.average_clustering(G_w, weight="weight"))
    except Exception:  # noqa: BLE001
        out["clustering"] = float("nan")

    return out


def _per_graph_values(
    graphs: Sequence[SubjectRunGraph], metric_name: str, seed: int = 42
) -> np.ndarray:
    vals = np.array([compute_metrics(g, seed=seed).get(metric_name, np.nan) for g in graphs])
    return vals[~np.isnan(vals)]


# --------------------------------------------------------------------------- #
# Stability analyses
# --------------------------------------------------------------------------- #
def bootstrap_metric(
    graphs: Sequence[SubjectRunGraph],
    metric_name: str,
    n_resamples: int = 500,
    random_state: int | None = 42,
    cv_warning_threshold: float = 0.20,
) -> GraphMetricSummary:
    """Bootstrap the across-graph mean of ``metric_name``."""
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric: {metric_name!r}; expected one of {METRICS}")
    if not graphs:
        raise ValueError("At least one graph required")

    values = _per_graph_values(graphs, metric_name, seed=int(random_state or 0))
    if values.size == 0:
        raise ValueError(f"All {metric_name} values were NaN")

    rng = np.random.default_rng(random_state)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = float(np.mean(sample))

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1) if values.size > 1 else 0.0)
    cv = float(std / mean) if mean not in (0.0, -0.0) else float("nan")
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])

    notes = "high_CV" if (not np.isnan(cv) and cv > cv_warning_threshold) else ""

    return GraphMetricSummary(
        metric_name=metric_name,
        density_threshold=graphs[0].density_threshold,
        mean=mean,
        std=std,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        coefficient_of_variation=cv,
        n_resamples=n_resamples,
        method="bootstrap",
        notes=notes,
    )


def leave_one_run_out(
    graphs: Sequence[SubjectRunGraph],
    metric_name: str,
    cv_warning_threshold: float = 0.20,
    random_state: int | None = 42,
) -> GraphMetricSummary:
    """Recompute the cross-graph mean omitting each graph in turn."""
    if metric_name not in METRICS:
        raise ValueError(f"Unknown metric: {metric_name!r}; expected one of {METRICS}")
    if len(graphs) < 2:
        raise ValueError("Need at least 2 graphs for leave-one-out")

    values = _per_graph_values(graphs, metric_name, seed=int(random_state or 0))
    if values.size < 2:
        raise ValueError(f"Need at least 2 non-NaN {metric_name} values")

    loo_means = np.array(
        [np.mean(np.delete(values, i)) for i in range(values.size)], dtype=float
    )
    mean = float(np.mean(loo_means))
    std = float(np.std(loo_means, ddof=1) if loo_means.size > 1 else 0.0)
    cv = float(std / mean) if mean not in (0.0, -0.0) else float("nan")

    notes_parts = ["leave_one_run_out"]
    if not np.isnan(cv) and cv > cv_warning_threshold:
        notes_parts.append("high_CV")

    return GraphMetricSummary(
        metric_name=metric_name,
        density_threshold=graphs[0].density_threshold,
        mean=mean,
        std=std,
        ci_low=float(np.min(loo_means)),
        ci_high=float(np.max(loo_means)),
        coefficient_of_variation=cv,
        n_resamples=int(values.size),
        method="leave_one_run_out",
        notes="|".join(notes_parts),
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run(config: Dict[str, Any] | str | Path) -> GraphStabilityResult:
    """Entry point. Reads config.graph.stability.* keys."""
    import yaml

    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = yaml.safe_load(f)
    graph_cfg = (config.get("graph") or {}).get("stability", {}) if config else {}
    paths = config.get("paths", {}) if config else {}

    input_dir = Path(paths["graph_input_dir"])
    output_dir = Path(paths["output_root"])

    thresholds = list(graph_cfg.get("density_thresholds", [0.10, 0.15, 0.20]))
    primary = float(graph_cfg.get("primary_threshold", thresholds[0]))
    metrics_requested = list(graph_cfg.get("metrics", METRICS))
    n_bootstrap = int(graph_cfg.get("n_bootstrap", 500))
    do_loro = bool(graph_cfg.get("leave_one_run_out", True))
    cv_warn = float(graph_cfg.get("cv_warning_threshold", 0.20))

    fcs = _load_fc_matrices(input_dir)
    result = GraphStabilityResult(
        primary_threshold=primary,
        thresholds_considered=thresholds,
    )
    for d in thresholds:
        graphs = [
            SubjectRunGraph(
                subject_id=fc.subject_id,
                run_id=fc.run_id,
                adjacency=threshold_adjacency(fc.fc_matrix, d),
                density_threshold=d,
            )
            for fc in fcs
        ]
        for metric in metrics_requested:
            result.summaries.append(
                bootstrap_metric(
                    graphs,
                    metric,
                    n_resamples=n_bootstrap,
                    cv_warning_threshold=cv_warn,
                )
            )
            if do_loro and len(graphs) >= 2:
                result.summaries.append(
                    leave_one_run_out(
                        graphs, metric, cv_warning_threshold=cv_warn
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_graph_summary(result, output_dir, output_csv=graph_cfg.get("output_csv"))
    return result


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def _load_fc_matrices(input_dir: Path):
    # Reuse fc_reproducibility's loader so layout conventions stay in one place.
    from fmri_pipeline.reproducibility.fc_reproducibility import load_run_fcs

    return load_run_fcs(input_dir)


def write_graph_summary(
    result: GraphStabilityResult,
    output_dir: Path,
    output_csv: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (output_csv or "graph_metrics_bootstrap.csv")
    rows = [asdict(s) for s in result.summaries]
    fieldnames = [
        "metric_name", "density_threshold", "mean", "std",
        "ci_low", "ci_high", "coefficient_of_variation",
        "n_resamples", "method", "notes",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return csv_path
