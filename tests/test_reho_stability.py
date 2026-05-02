"""Synthetic-data tests for fmri_pipeline.reproducibility.reho_stability.

The test design mirrors test_fc_reproducibility.py: build small datasets
where the within > between contrast is true (or false) by construction,
and assert that the analysis returns the right verdict.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.reproducibility.reho_stability import (  # noqa: E402
    ReHoStabilityResult,
    SubjectRunReHo,
    compute_between_subject_similarity,
    compute_similarity_matrix,
    compute_within_subject_similarity,
    load_run_reho_vectors,
    run,
    summarize,
    write_outputs,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_synthetic_rehos(
    n_subjects: int = 5,
    n_runs: int = 3,
    n_rois: int = 60,
    noise: float = 0.2,
    seed: int = 42,
):
    """Each subject has a unique ROI ReHo template; each run adds noise."""
    rng = np.random.default_rng(seed)
    templates = [rng.standard_normal(n_rois) for _ in range(n_subjects)]
    out = []
    for sid, t in enumerate(templates):
        for rid in range(n_runs):
            v = t + rng.standard_normal(n_rois) * noise
            out.append(
                SubjectRunReHo(
                    subject_id=f"S{sid:02d}",
                    run_id=f"{rid + 1}",
                    reho_roi_vector=v,
                )
            )
    return out


def _make_null_rehos(n_labels: int = 5, n_runs: int = 3, n_rois: int = 60, seed: int = 7):
    """No subject-level structure; subject labels assigned round-robin."""
    rng = np.random.default_rng(seed)
    out = []
    total = n_labels * n_runs
    for i in range(total):
        out.append(
            SubjectRunReHo(
                subject_id=f"S{i % n_labels:02d}",
                run_id=f"{(i // n_labels) + 1}",
                reho_roi_vector=rng.standard_normal(n_rois),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Similarity matrix
# --------------------------------------------------------------------------- #
class TestSimilarityMatrix:
    def test_diagonal_is_one(self) -> None:
        rehos = _make_synthetic_rehos(n_subjects=3, n_runs=2)
        mat, _ = compute_similarity_matrix(rehos)
        np.testing.assert_allclose(np.diag(mat), 1.0)

    def test_symmetric(self) -> None:
        rehos = _make_synthetic_rehos()
        mat, _ = compute_similarity_matrix(rehos)
        np.testing.assert_allclose(mat, mat.T)

    def test_shape_and_index_match_runs(self) -> None:
        rehos = _make_synthetic_rehos(n_subjects=4, n_runs=3)
        mat, idx = compute_similarity_matrix(rehos)
        assert mat.shape == (12, 12)
        assert len(idx) == 12
        # Index should be (subject_id, run_id) tuples in original order.
        assert idx[0] == (rehos[0].subject_id, rehos[0].run_id)

    def test_values_in_range(self) -> None:
        rehos = _make_synthetic_rehos()
        mat, _ = compute_similarity_matrix(rehos)
        # Off-diagonal correlations must be in [-1, 1].
        off = mat[~np.eye(mat.shape[0], dtype=bool)]
        assert off.min() >= -1.0 - 1e-9
        assert off.max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------- #
# Within-subject similarity
# --------------------------------------------------------------------------- #
class TestWithinSubjectSimilarity:
    def test_skips_single_run_subjects(self) -> None:
        rng = np.random.default_rng(0)
        only_one = SubjectRunReHo("solo", "1", rng.standard_normal(20))
        out = compute_within_subject_similarity([only_one])
        assert "solo" not in out

    def test_pair_counts(self) -> None:
        rehos = _make_synthetic_rehos(n_subjects=2, n_runs=4)
        out = compute_within_subject_similarity(rehos)
        for sims in out.values():
            assert len(sims) == 6  # C(4, 2)


# --------------------------------------------------------------------------- #
# Between-subject similarity
# --------------------------------------------------------------------------- #
class TestBetweenSubjectSimilarity:
    def test_matched_run_index_count(self) -> None:
        # 3 subjects x 2 matched runs => 3 pairs of subjects * 2 = 6 sims
        rehos = _make_synthetic_rehos(n_subjects=3, n_runs=2)
        sims = compute_between_subject_similarity(rehos, match_run_index=True)
        assert len(sims) == 6

    def test_all_pairs_count(self) -> None:
        rehos = _make_synthetic_rehos(n_subjects=3, n_runs=2)
        sims = compute_between_subject_similarity(rehos, match_run_index=False)
        # 3 subject-pairs * 2*2 cross-run = 12
        assert len(sims) == 12


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
class TestSummarize:
    def test_within_exceeds_between_on_structured_data(self) -> None:
        rehos = _make_synthetic_rehos(n_subjects=6, n_runs=3, noise=0.15)
        sim_mat, idx = compute_similarity_matrix(rehos)
        within = compute_within_subject_similarity(rehos)
        between = compute_between_subject_similarity(rehos)
        result = summarize(
            within,
            between,
            similarity_matrix=sim_mat,
            run_index=idx,
            n_subjects=6,
            n_bootstrap=200,
            random_state=0,
        )
        assert result.within_mean > result.between_mean
        assert result.gap_mean > 0
        assert result.p_value < 0.05
        assert result.effect_size > 0

    def test_null_case_is_not_strongly_significant(self) -> None:
        rehos = _make_null_rehos()
        sim_mat, idx = compute_similarity_matrix(rehos)
        within = compute_within_subject_similarity(rehos)
        between = compute_between_subject_similarity(rehos)
        result = summarize(
            within,
            between,
            similarity_matrix=sim_mat,
            run_index=idx,
            n_subjects=5,
            n_bootstrap=200,
            random_state=0,
        )
        assert result.p_value > 0.01

    def test_bootstrap_ci_brackets_gap_mean(self) -> None:
        rehos = _make_synthetic_rehos()
        sim_mat, idx = compute_similarity_matrix(rehos)
        within = compute_within_subject_similarity(rehos)
        between = compute_between_subject_similarity(rehos)
        result = summarize(
            within, between,
            similarity_matrix=sim_mat, run_index=idx,
            n_subjects=5, n_bootstrap=500, random_state=0,
        )
        assert result.gap_ci_low <= result.gap_mean <= result.gap_ci_high

    def test_perfect_replicas_yield_unit_within(self) -> None:
        """Two identical runs per subject -> within similarity == 1 (clipped)."""
        rng = np.random.default_rng(0)
        rehos = []
        for sid in range(3):
            v = rng.standard_normal(40)
            for rid in range(2):
                rehos.append(SubjectRunReHo(f"S{sid}", f"{rid + 1}", v.copy()))
        within = compute_within_subject_similarity(rehos)
        for vs in within.values():
            for r in vs:
                assert r > 0.999

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            summarize({}, [], n_bootstrap=10)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class TestIO:
    def test_load_run_reho_vectors_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(11)
        for s in range(2):
            for r in range(2):
                np.save(
                    tmp_path / f"sub-{s:02d}_run-{r + 1}_reho.npy",
                    rng.standard_normal(15),
                )
        rehos = load_run_reho_vectors(tmp_path)
        assert len(rehos) == 4
        assert {rh.subject_id for rh in rehos} == {"00", "01"}
        assert {rh.run_id for rh in rehos} == {"1", "2"}

    def test_load_rejects_2d_arrays(self, tmp_path: Path) -> None:
        np.save(tmp_path / "sub-01_run-1_reho.npy", np.zeros((10, 5)))
        with pytest.raises(ValueError):
            load_run_reho_vectors(tmp_path)

    def test_load_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run_reho_vectors(tmp_path / "missing")

    def test_write_outputs_creates_artifacts(self, tmp_path: Path) -> None:
        rehos = _make_synthetic_rehos(n_subjects=3, n_runs=2)
        sim_mat, idx = compute_similarity_matrix(rehos)
        within = compute_within_subject_similarity(rehos)
        between = compute_between_subject_similarity(rehos)
        result = summarize(
            within, between,
            similarity_matrix=sim_mat, run_index=idx,
            n_subjects=3, n_bootstrap=50, random_state=0,
        )
        paths = write_outputs(result, tmp_path)
        assert paths["matrix"].exists()
        assert paths["csv"].exists()
        assert paths["json"].exists()
        assert paths["index"].exists()

        # Matrix shape must match number of runs.
        assert np.load(paths["matrix"]).shape == (6, 6)
        # CSV must have header + one row per dataclass field (minus arrays).
        with paths["csv"].open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["metric", "value"]
        # similarity_matrix and run_index should not appear as rows.
        keys_written = {r[0] for r in rows[1:]}
        assert "similarity_matrix" not in keys_written
        assert "run_index" not in keys_written


# --------------------------------------------------------------------------- #
# End-to-end run() entry point
# --------------------------------------------------------------------------- #
class TestRunEntryPoint:
    def test_run_end_to_end(self, tmp_path: Path) -> None:
        # Build synthetic data on disk.
        input_dir = tmp_path / "reho_in"
        output_dir = tmp_path / "reproducibility"
        input_dir.mkdir(parents=True)
        rng = np.random.default_rng(1)
        for sid in range(4):
            template = rng.standard_normal(50)
            for rid in range(3):
                noisy = template + rng.standard_normal(50) * 0.15
                np.save(input_dir / f"sub-{sid:02d}_run-{rid + 1}_reho.npy", noisy)

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "project": {"random_seed": 0},
                    "paths": {
                        "reho_input_dir": str(input_dir),
                        "output_root": str(output_dir),
                    },
                    "reho": {
                        "match_between_by_run_index": True,
                        "n_bootstrap": 100,
                    },
                }
            )
        )

        result = run(cfg_path)

        assert isinstance(result, ReHoStabilityResult)
        assert result.gap_mean > 0
        assert (output_dir / "reho_summary.csv").exists()
        assert (output_dir / "reho_similarity_matrix.npy").exists()
