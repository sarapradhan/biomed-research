"""Tests for fmri_pipeline.reproducibility.scorecard."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
import yaml

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.reproducibility.scorecard import (  # noqa: E402
    ScorecardRow,
    build_scorecard,
    render_csv,
    render_markdown,
    run,
    write_outputs,
)


# --------------------------------------------------------------------------- #
# Fixture helpers — write the per-module CSVs the scorecard will read.
# --------------------------------------------------------------------------- #
def _write_metric_value_csv(path: Path, kvs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in kvs.items():
            w.writerow([k, v])


def _write_graph_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "metric_name", "density_threshold", "mean", "std",
        "ci_low", "ci_high", "coefficient_of_variation",
        "n_resamples", "method", "notes",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _good_reports(reports_dir: Path) -> None:
    """Write a complete set of 'pass' summary files."""
    reports_dir.mkdir(parents=True, exist_ok=True)

    _write_metric_value_csv(
        reports_dir / "fc_within_vs_between.csv",
        {
            "within_mean": 0.62, "within_std": 0.10, "between_mean": 0.30,
            "between_std": 0.12, "gap_mean": 0.32,
            "gap_ci_low": 0.20, "gap_ci_high": 0.44,
            "n_subjects": 12, "n_within_pairs": 36, "n_between_pairs": 66,
            "test_statistic": 250.0, "p_value": 0.0001, "effect_size": 1.4,
        },
    )
    _write_metric_value_csv(
        reports_dir / "reho_summary.csv",
        {
            "within_mean": 0.55, "within_std": 0.08, "between_mean": 0.20,
            "between_std": 0.11, "gap_mean": 0.35,
            "gap_ci_low": 0.22, "gap_ci_high": 0.48, "n_subjects": 12,
            "n_runs": 36, "n_within_pairs": 36, "n_between_pairs": 66,
            "test_statistic": 220.0, "p_value": 0.0005, "effect_size": 1.2,
        },
    )
    _write_metric_value_csv(
        reports_dir / "ica_stability_seeds.csv",
        {
            "sweep": "seeds", "k_components": 20, "robust_threshold": 0.7,
            "n_runs": 5, "pairwise_runs_considered": 10,
            "mean_matched_correlation": 0.83,
            "std_matched_correlation": 0.07,
            "median_matched_correlation": 0.84,
            "n_robust_components": 16, "reference_tag": "seed-1",
        },
    )
    _write_metric_value_csv(
        reports_dir / "ica_stability_lorocv.csv",
        {
            "sweep": "run_subsets", "k_components": 20, "robust_threshold": 0.7,
            "n_runs": 4, "pairwise_runs_considered": 6,
            "mean_matched_correlation": 0.78,
            "std_matched_correlation": 0.09,
            "median_matched_correlation": 0.80,
            "n_robust_components": 14, "reference_tag": "loro-1",
        },
    )
    _write_graph_csv(
        reports_dir / "graph_metrics_bootstrap.csv",
        [
            {
                "metric_name": "modularity", "density_threshold": 0.15,
                "mean": 0.34, "std": 0.04, "ci_low": 0.30, "ci_high": 0.38,
                "coefficient_of_variation": 0.12, "n_resamples": 500,
                "method": "bootstrap", "notes": "",
            },
            {
                "metric_name": "modularity", "density_threshold": 0.10,
                "mean": 0.32, "std": 0.05, "ci_low": 0.28, "ci_high": 0.36,
                "coefficient_of_variation": 0.16, "n_resamples": 500,
                "method": "bootstrap", "notes": "",
            },
            {
                "metric_name": "global_efficiency", "density_threshold": 0.15,
                "mean": 0.62, "std": 0.03, "ci_low": 0.59, "ci_high": 0.65,
                "coefficient_of_variation": 0.05, "n_resamples": 500,
                "method": "bootstrap", "notes": "",
            },
        ],
    )
    (reports_dir / "dfc_sensitivity.json").write_text(json.dumps({
        "window_sizes": [20, 30, 40],
        "n_runs": 4,
        "per_window_variability_mean": {"20": 0.48, "30": 0.42, "40": 0.39},
        "per_window_variability_std": {"20": 0.01, "30": 0.01, "40": 0.01},
        "pairwise_ari_mean": {"20-30": 0.62, "20-40": 0.49, "30-40": 0.72},
        "pairwise_ari_std": {"20-30": 0.04, "20-40": 0.02, "30-40": 0.10},
    }))
    _write_metric_value_csv(
        reports_dir / "network_anchor_summary.csv",
        {
            "n_rois": 100, "n_networks": 7,
            "within_mean": 0.42, "between_mean": 0.10,
            "gap_mean": 0.32, "p_value": 0.001, "n_permutations": 1000,
            "modularity_q": 0.45,
        },
    )


# --------------------------------------------------------------------------- #
# build_scorecard
# --------------------------------------------------------------------------- #
class TestBuildScorecard:
    def test_all_pass_when_inputs_are_good(self, tmp_path: Path) -> None:
        _good_reports(tmp_path)
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.15)
        assert len(rows) == 7
        # Areas appear in the canonical order.
        assert [r.area for r in rows] == [
            "Reproducibility of FC",
            "ReHo stability",
            "ICA stability",
            "ICA stability",
            "Graph metric stability",
            "Dynamic FC robustness",
            "Static FC plausibility",
        ]
        statuses = {f"{r.area}::{r.check}": r.pass_status for r in rows}
        assert statuses["Reproducibility of FC::Within-subject run similarity"] == "OK"
        assert statuses["ReHo stability::Cross-run similarity"] == "OK"
        assert statuses["ICA stability::Component recovery across seeds"] == "OK"
        assert statuses["ICA stability::Component recovery across run subsets"] == "OK"
        assert statuses["Graph metric stability::Bootstrap consistency"] == "OK"
        # dFC is "describe, don't gate".
        assert statuses["Dynamic FC robustness::Multi-window sensitivity"] == "n/a"
        assert statuses["Static FC plausibility::Canonical network organization"] == "OK"

    def test_missing_file_returns_n_a_row(self, tmp_path: Path) -> None:
        # Don't write any inputs -> every row should be "(missing)" / "n/a".
        rows = build_scorecard(tmp_path)
        assert all(r.pass_status == "n/a" for r in rows)
        assert all("missing" in r.result for r in rows)

    def test_failing_fc_marked_fail(self, tmp_path: Path) -> None:
        _good_reports(tmp_path)
        # Overwrite FC with a non-significant gap.
        _write_metric_value_csv(
            tmp_path / "fc_within_vs_between.csv",
            {
                "within_mean": 0.30, "between_mean": 0.31, "gap_mean": -0.01,
                "gap_ci_low": -0.05, "gap_ci_high": 0.03, "p_value": 0.50,
                "effect_size": -0.05, "n_subjects": 5,
            },
        )
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.15)
        fc_row = next(r for r in rows if r.area == "Reproducibility of FC")
        assert fc_row.pass_status == "FAIL"

    def test_high_cv_graph_marked_fail(self, tmp_path: Path) -> None:
        _good_reports(tmp_path)
        # Bump CV above the warning threshold for the primary metric+threshold.
        _write_graph_csv(
            tmp_path / "graph_metrics_bootstrap.csv",
            [
                {
                    "metric_name": "modularity", "density_threshold": 0.15,
                    "mean": 0.34, "std": 0.4, "ci_low": 0.0, "ci_high": 0.7,
                    "coefficient_of_variation": 0.5, "n_resamples": 500,
                    "method": "bootstrap", "notes": "high_CV",
                },
            ],
        )
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.15)
        graph_row = next(r for r in rows if r.area == "Graph metric stability")
        assert graph_row.pass_status == "FAIL"

    def test_low_robust_ratio_ica_marked_fail(self, tmp_path: Path) -> None:
        _good_reports(tmp_path)
        _write_metric_value_csv(
            tmp_path / "ica_stability_seeds.csv",
            {
                "sweep": "seeds", "k_components": 20, "robust_threshold": 0.7,
                "n_runs": 5, "pairwise_runs_considered": 10,
                "mean_matched_correlation": 0.4, "std_matched_correlation": 0.1,
                "median_matched_correlation": 0.4,
                "n_robust_components": 5, "reference_tag": "seed-1",
            },
        )
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.15)
        ica_row = next(
            r for r in rows
            if r.area == "ICA stability" and "seeds" in r.check
        )
        assert ica_row.pass_status == "FAIL"

    def test_graph_threshold_filter_falls_back(self, tmp_path: Path) -> None:
        """If primary_threshold doesn't match any row, we should still get a result."""
        _good_reports(tmp_path)
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.99)
        graph_row = next(r for r in rows if r.area == "Graph metric stability")
        # Falls through to the first bootstrap+modularity row -> threshold=0.15.
        assert "modularity" in graph_row.result


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
class TestRenderers:
    def test_markdown_has_header_and_one_row_each(self, tmp_path: Path) -> None:
        _good_reports(tmp_path)
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.15)
        md = render_markdown(rows)
        # Sanity-check structure.
        assert md.startswith("# Validation Scorecard")
        assert md.count("\n|") >= len(rows) + 2  # header + sep + rows
        assert "Reproducibility of FC" in md
        assert "Static FC plausibility" in md

    def test_pipe_chars_in_results_are_escaped(self, tmp_path: Path) -> None:
        rows = [ScorecardRow("A | B", "C | D", "x | y", "OK", "f.csv")]
        md = render_markdown(rows)
        assert r"\|" in md

    def test_render_csv_roundtrip(self, tmp_path: Path) -> None:
        rows = [
            ScorecardRow("X", "Y", "result", "OK", "x.csv"),
            ScorecardRow("X", "Z", "other", "n/a", "x.csv"),
        ]
        path = render_csv(rows, tmp_path / "out.csv")
        assert path.exists()
        with path.open() as f:
            data = list(csv.DictReader(f))
        assert len(data) == 2
        assert data[0]["pass_status"] == "OK"

    def test_write_outputs_creates_both(self, tmp_path: Path) -> None:
        _good_reports(tmp_path)
        rows = build_scorecard(tmp_path, primary_graph_threshold=0.15)
        out_dir = tmp_path / "scorecards"
        paths = write_outputs(rows, out_dir)
        assert paths["markdown"].exists()
        assert paths["csv"].exists()


# --------------------------------------------------------------------------- #
# run() entry point
# --------------------------------------------------------------------------- #
class TestRunEntryPoint:
    def test_run_with_yaml_path(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports"
        _good_reports(reports_dir)

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "paths": {"output_root": str(reports_dir)},
                    "scorecard": {
                        "graph_primary_metric": "modularity",
                    },
                    "graph": {
                        "stability": {
                            "primary_threshold": 0.15,
                            "cv_warning_threshold": 0.20,
                        }
                    },
                }
            )
        )
        paths = run(cfg)
        assert paths["markdown"].exists()
        assert paths["csv"].exists()
        text = paths["markdown"].read_text()
        assert "Validation Scorecard" in text
        assert "Reproducibility of FC" in text

    def test_run_with_dict_config(self, tmp_path: Path) -> None:
        reports_dir = tmp_path / "reports"
        _good_reports(reports_dir)
        paths = run(
            {
                "paths": {"output_root": str(reports_dir)},
                "scorecard": {"graph_primary_metric": "modularity"},
                "graph": {"stability": {"primary_threshold": 0.15}},
            }
        )
        assert paths["markdown"].exists()
