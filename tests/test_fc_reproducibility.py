"""Synthetic-data tests for fmri_pipeline.reproducibility.fc_reproducibility.

No neuroimaging files are required. Each test builds a controlled FC
dataset where the within > between relationship is known by construction.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.reproducibility.fc_reproducibility import (  # noqa: E402
    ReproducibilityResult,
    SubjectRunFC,
    compute_between_subject_similarity,
    compute_within_subject_similarity,
    load_run_fcs,
    summarize,
    write_summary,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _symmetric_zero_diag(m: np.ndarray) -> np.ndarray:
    out = (m + m.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def _make_synthetic_run_fcs(
    n_subjects: int = 5,
    n_runs: int = 3,
    n_rois: int = 20,
    noise: float = 0.15,
    seed: int = 42,
):
    """Each subject has a unique template FC; each run adds Gaussian noise.

    By construction: within-subject run-pair similarity > between-subject
    run-pair similarity.
    """
    rng = np.random.default_rng(seed)
    templates = [
        _symmetric_zero_diag(rng.standard_normal((n_rois, n_rois)))
        for _ in range(n_subjects)
    ]
    run_fcs = []
    for sid, t in enumerate(templates):
        for rid in range(n_runs):
            noise_mat = _symmetric_zero_diag(
                rng.standard_normal((n_rois, n_rois)) * noise
            )
            run_fcs.append(
                SubjectRunFC(
                    subject_id=f"S{sid:02d}",
                    run_id=f"{rid + 1}",
                    fc_matrix=t + noise_mat,
                )
            )
    return run_fcs


def _make_null_run_fcs(
    n_labels: int = 5, n_runs: int = 3, n_rois: int = 20, seed: int = 7
):
    """No subject-level structure: every matrix drawn independently.

    Subject labels are assigned round-robin, so within-subject and
    between-subject similarity should be indistinguishable.
    """
    rng = np.random.default_rng(seed)
    run_fcs = []
    total = n_labels * n_runs
    for i in range(total):
        m = _symmetric_zero_diag(rng.standard_normal((n_rois, n_rois)))
        run_fcs.append(
            SubjectRunFC(
                subject_id=f"S{i % n_labels:02d}",
                run_id=f"{(i // n_labels) + 1}",
                fc_matrix=m,
            )
        )
    return run_fcs


# --------------------------------------------------------------------------- #
# Upper triangle extraction
# --------------------------------------------------------------------------- #
class TestUpperTriangle:
    def test_length_is_n_choose_2(self) -> None:
        rng = np.random.default_rng(0)
        m = _symmetric_zero_diag(rng.standard_normal((10, 10)))
        rf = SubjectRunFC("s1", "1", m)
        assert rf.upper_triangle().shape == (45,)

    def test_excludes_diagonal(self) -> None:
        m = np.eye(5) * 99.0 + np.ones((5, 5))
        rf = SubjectRunFC("s1", "1", m)
        vec = rf.upper_triangle()
        assert not np.any(vec == 99.0 + 1.0)


# --------------------------------------------------------------------------- #
# Within-subject similarity
# --------------------------------------------------------------------------- #
class TestWithinSubjectSimilarity:
    def test_skips_subjects_with_single_run(self) -> None:
        rng = np.random.default_rng(1)
        only_one = SubjectRunFC(
            "s-only", "1", _symmetric_zero_diag(rng.standard_normal((5, 5)))
        )
        out = compute_within_subject_similarity([only_one])
        assert "s-only" not in out

    def test_all_multi_run_subjects_present(self) -> None:
        fcs = _make_synthetic_run_fcs(n_subjects=4, n_runs=2)
        out = compute_within_subject_similarity(fcs)
        assert set(out.keys()) == {"S00", "S01", "S02", "S03"}

    def test_pair_counts(self) -> None:
        fcs = _make_synthetic_run_fcs(n_subjects=3, n_runs=4)
        out = compute_within_subject_similarity(fcs)
        for sid, sims in out.items():
            assert len(sims) == 6  # 4 choose 2

    def test_correlations_in_range(self) -> None:
        fcs = _make_synthetic_run_fcs()
        out = compute_within_subject_similarity(fcs)
        for sims in out.values():
            for r in sims:
                assert -1.0 <= r <= 1.0


# --------------------------------------------------------------------------- #
# Between-subject similarity
# --------------------------------------------------------------------------- #
class TestBetweenSubjectSimilarity:
    def test_matched_run_index_count(self) -> None:
        # 3 subjects x 2 matched run indices = 3 pairs * 2 = 6 sims
        fcs = _make_synthetic_run_fcs(n_subjects=3, n_runs=2)
        sims = compute_between_subject_similarity(fcs, match_run_index=True)
        assert len(sims) == 6

    def test_all_pairs_count(self) -> None:
        fcs = _make_synthetic_run_fcs(n_subjects=3, n_runs=2)
        sims = compute_between_subject_similarity(fcs, match_run_index=False)
        # 3 subject pairs, each has 2 x 2 cross-run pairs = 12
        assert len(sims) == 12

    def test_correlations_in_range(self) -> None:
        fcs = _make_synthetic_run_fcs()
        sims = compute_between_subject_similarity(fcs)
        for r in sims:
            assert -1.0 <= r <= 1.0


# --------------------------------------------------------------------------- #
# Summary statistics
# --------------------------------------------------------------------------- #
class TestSummarize:
    def test_within_exceeds_between_on_structured_data(self) -> None:
        fcs = _make_synthetic_run_fcs(n_subjects=6, n_runs=3, noise=0.1)
        within = compute_within_subject_similarity(fcs)
        between = compute_between_subject_similarity(fcs)
        result = summarize(within, between, n_bootstrap=200, random_state=0)
        assert result.within_mean > result.between_mean
        assert result.gap_mean > 0
        assert result.p_value < 0.05
        assert result.effect_size > 0

    def test_bootstrap_ci_brackets_gap_mean(self) -> None:
        fcs = _make_synthetic_run_fcs()
        within = compute_within_subject_similarity(fcs)
        between = compute_between_subject_similarity(fcs)
        result = summarize(within, between, n_bootstrap=500, random_state=0)
        # The bootstrap CI should contain or narrowly miss the point estimate;
        # for stable synthetic data it should contain it.
        assert result.gap_ci_low <= result.gap_mean <= result.gap_ci_high

    def test_null_case_does_not_pass_significance(self) -> None:
        fcs = _make_null_run_fcs()
        within = compute_within_subject_similarity(fcs)
        between = compute_between_subject_similarity(fcs)
        result = summarize(within, between, n_bootstrap=200, random_state=0)
        # Not a strict test (single draw) but p should not be strongly significant
        assert result.p_value > 0.01

    def test_n_subjects_matches_within_keys(self) -> None:
        fcs = _make_synthetic_run_fcs(n_subjects=4, n_runs=3)
        within = compute_within_subject_similarity(fcs)
        between = compute_between_subject_similarity(fcs)
        result = summarize(within, between, n_bootstrap=100, random_state=0)
        assert result.n_subjects == 4

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError):
            summarize({}, [], n_bootstrap=10)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class TestIO:
    def test_load_run_fcs_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(11)
        for s in range(2):
            for r in range(2):
                m = _symmetric_zero_diag(rng.standard_normal((8, 8)))
                np.save(tmp_path / f"sub-{s:02d}_run-{r + 1}_fc.npy", m)
        fcs = load_run_fcs(tmp_path)
        assert len(fcs) == 4
        assert {fc.subject_id for fc in fcs} == {"00", "01"}
        assert {fc.run_id for fc in fcs} == {"1", "2"}

    def test_load_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run_fcs(tmp_path / "does-not-exist")

    def test_write_summary_creates_csv_and_json(self, tmp_path: Path) -> None:
        result = ReproducibilityResult(
            within_mean=0.6, within_std=0.1, between_mean=0.3, between_std=0.1,
            gap_mean=0.3, gap_ci_low=0.2, gap_ci_high=0.4,
            n_subjects=4, n_within_pairs=12, n_between_pairs=6,
            test_statistic=10.0, p_value=0.001, effect_size=1.2,
        )
        csv_path = write_summary(result, tmp_path)
        assert csv_path.exists()
        assert (tmp_path / (csv_path.stem + ".json")).exists()
        with csv_path.open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["metric", "value"]
        assert any(row[0] == "within_mean" for row in rows[1:])
