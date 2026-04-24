"""Synthetic-data tests for fmri_pipeline.reproducibility.graph_stability.

Build FC matrices with known block structure, threshold them, and verify
that modularity, efficiency, and clustering all fall in expected ranges
and that the bootstrap / leave-one-run-out summaries are well formed.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.reproducibility.graph_stability import (  # noqa: E402
    GraphMetricSummary,
    GraphStabilityResult,
    SubjectRunGraph,
    bootstrap_metric,
    compute_metrics,
    leave_one_run_out,
    threshold_adjacency,
    write_graph_summary,
)

nx = pytest.importorskip("networkx")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _block_fc(n_rois: int = 20, noise: float = 0.03, seed: int = 0) -> np.ndarray:
    """Two-community block FC with positive within-block edges and noise."""
    rng = np.random.default_rng(seed)
    half = n_rois // 2
    template = np.full((n_rois, n_rois), -0.2)
    template[:half, :half] = 0.8
    template[half:, half:] = 0.8
    noise_mat = rng.standard_normal((n_rois, n_rois)) * noise
    fc = template + noise_mat
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)
    return fc


def _make_graphs(n: int = 5, density: float = 0.2, n_rois: int = 20) -> list:
    return [
        SubjectRunGraph(
            subject_id=f"s{i}",
            run_id=str(i + 1),
            adjacency=threshold_adjacency(_block_fc(n_rois=n_rois, seed=i), density),
            density_threshold=density,
        )
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# threshold_adjacency
# --------------------------------------------------------------------------- #
class TestThresholdAdjacency:
    def test_density_close_to_target(self) -> None:
        fc = _block_fc(n_rois=30)
        adj = threshold_adjacency(fc, density=0.15)
        n_edges_kept = int(np.sum(np.abs(adj[np.triu_indices(30, k=1)]) > 0))
        total = 30 * 29 // 2
        assert abs(n_edges_kept / total - 0.15) < 0.03

    def test_diagonal_is_zero(self) -> None:
        adj = threshold_adjacency(_block_fc(), density=0.2)
        assert np.all(np.diag(adj) == 0)

    def test_is_symmetric(self) -> None:
        adj = threshold_adjacency(_block_fc(), density=0.2)
        assert np.allclose(adj, adj.T)

    def test_rejects_bad_density(self) -> None:
        with pytest.raises(ValueError):
            threshold_adjacency(_block_fc(), density=1.5)
        with pytest.raises(ValueError):
            threshold_adjacency(_block_fc(), density=0.0)

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError):
            threshold_adjacency(np.zeros((10, 5)), density=0.2)


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
class TestComputeMetrics:
    def test_returns_all_three(self) -> None:
        g = _make_graphs(n=1)[0]
        m = compute_metrics(g)
        assert set(m.keys()) == {"modularity", "global_efficiency", "clustering"}

    def test_modularity_positive_for_block_structure(self) -> None:
        fc = _block_fc(n_rois=30, noise=0.02, seed=3)
        adj = threshold_adjacency(fc, density=0.3)
        g = SubjectRunGraph("s", "1", adj, 0.3)
        m = compute_metrics(g)
        assert m["modularity"] > 0.1

    def test_efficiency_in_unit_interval(self) -> None:
        g = _make_graphs(n=1)[0]
        m = compute_metrics(g)
        assert 0.0 <= m["global_efficiency"] <= 1.0

    def test_clustering_nonnegative(self) -> None:
        g = _make_graphs(n=1)[0]
        m = compute_metrics(g)
        assert m["clustering"] >= 0.0


# --------------------------------------------------------------------------- #
# bootstrap_metric
# --------------------------------------------------------------------------- #
class TestBootstrapMetric:
    def test_ci_brackets_mean(self) -> None:
        graphs = _make_graphs(n=6)
        s = bootstrap_metric(graphs, "clustering", n_resamples=100, random_state=0)
        assert s.ci_low <= s.mean <= s.ci_high
        assert s.method == "bootstrap"
        assert s.n_resamples == 100

    def test_unknown_metric_raises(self) -> None:
        graphs = _make_graphs(n=2)
        with pytest.raises(ValueError):
            bootstrap_metric(graphs, "not_a_metric")

    def test_empty_graphs_raises(self) -> None:
        with pytest.raises(ValueError):
            bootstrap_metric([], "clustering")

    def test_density_threshold_propagated(self) -> None:
        graphs = _make_graphs(n=4, density=0.18)
        s = bootstrap_metric(graphs, "modularity", n_resamples=50, random_state=0)
        assert s.density_threshold == 0.18

    def test_high_cv_flag(self) -> None:
        # Build graphs with very different structures so CV is large.
        graphs = []
        for i in range(5):
            fc = _block_fc(noise=0.2 + 0.4 * i, seed=i)
            graphs.append(
                SubjectRunGraph(f"s{i}", "1", threshold_adjacency(fc, 0.2), 0.2)
            )
        s = bootstrap_metric(
            graphs, "clustering", n_resamples=50, random_state=0,
            cv_warning_threshold=0.01,
        )
        assert "high_CV" in s.notes


# --------------------------------------------------------------------------- #
# leave_one_run_out
# --------------------------------------------------------------------------- #
class TestLeaveOneRunOut:
    def test_output_shape(self) -> None:
        graphs = _make_graphs(n=5)
        s = leave_one_run_out(graphs, "global_efficiency")
        assert s.metric_name == "global_efficiency"
        assert s.method == "leave_one_run_out"
        assert s.n_resamples == 5
        assert s.ci_low <= s.mean <= s.ci_high

    def test_raises_on_single_graph(self) -> None:
        graphs = _make_graphs(n=1)
        with pytest.raises(ValueError):
            leave_one_run_out(graphs, "clustering")

    def test_unknown_metric_raises(self) -> None:
        graphs = _make_graphs(n=3)
        with pytest.raises(ValueError):
            leave_one_run_out(graphs, "not_a_metric")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class TestWriteGraphSummary:
    def test_write_csv_has_all_rows(self, tmp_path: Path) -> None:
        summaries = [
            GraphMetricSummary(
                metric_name="clustering", density_threshold=0.2,
                mean=0.4, std=0.05, ci_low=0.3, ci_high=0.5,
                coefficient_of_variation=0.12, n_resamples=500,
                method="bootstrap", notes="",
            ),
            GraphMetricSummary(
                metric_name="modularity", density_threshold=0.2,
                mean=0.35, std=0.08, ci_low=0.2, ci_high=0.5,
                coefficient_of_variation=0.23, n_resamples=5,
                method="leave_one_run_out", notes="leave_one_run_out|high_CV",
            ),
        ]
        result = GraphStabilityResult(
            primary_threshold=0.2, thresholds_considered=[0.2], summaries=summaries
        )
        path = write_graph_summary(result, tmp_path)
        assert path.exists()
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert rows[0]["metric_name"] == "clustering"
        assert rows[1]["notes"] == "leave_one_run_out|high_CV"
