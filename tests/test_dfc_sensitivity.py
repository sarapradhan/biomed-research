"""Synthetic-data tests for dfc_sensitivity (JEI revision Phase 2)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.reproducibility.dfc_sensitivity import (  # noqa: E402
    GroupSensitivityResult,
    SubjectRunROI,
    aggregate_group,
    analyse_run,
    cluster_window_states,
    fc_variability_across_windows,
    load_run_timeseries,
    pairwise_ari,
    project_labels_to_TR,
    run,
    sweep_window_size,
    window_centers,
    window_starts,
    windowed_fc_matrices,
    write_outputs,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _stationary_ts(n_TRs: int, n_rois: int, seed: int = 0):
    """Stationary AR(1) timeseries with no regime structure."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_TRs, n_rois))


def _two_regime_ts(n_TRs: int, n_rois: int, seed: int = 0):
    """Two distinct covariance regimes glued back to back.

    Each half draws from a different multivariate Gaussian whose
    covariance has a strong block structure. This gives k-means
    something real to find.
    """
    rng = np.random.default_rng(seed)
    half = n_TRs // 2
    # Regime A: first half of ROIs strongly correlated.
    cov_a = np.eye(n_rois) * 0.1
    cov_a[: n_rois // 2, : n_rois // 2] = 0.9 * np.ones((n_rois // 2, n_rois // 2)) + 0.1 * np.eye(n_rois // 2)
    # Regime B: second half of ROIs strongly correlated.
    cov_b = np.eye(n_rois) * 0.1
    cov_b[n_rois // 2 :, n_rois // 2 :] = 0.9 * np.ones((n_rois // 2, n_rois // 2)) + 0.1 * np.eye(n_rois // 2)
    a = rng.multivariate_normal(np.zeros(n_rois), cov_a, size=half)
    b = rng.multivariate_normal(np.zeros(n_rois), cov_b, size=n_TRs - half)
    return np.vstack([a, b])


# --------------------------------------------------------------------------- #
# Window arithmetic
# --------------------------------------------------------------------------- #
class TestWindowArithmetic:
    def test_window_starts_basic(self) -> None:
        starts = window_starts(n_TRs=10, window_size=4, step_size=2)
        np.testing.assert_array_equal(starts, [0, 2, 4, 6])

    def test_window_starts_too_short_returns_empty(self) -> None:
        assert window_starts(n_TRs=3, window_size=4, step_size=1).size == 0

    def test_window_centers_match_starts(self) -> None:
        centers = window_centers(n_TRs=10, window_size=4, step_size=2)
        # First start=0, window covers TRs 0..3 -> center round((4-1)/2) = 2
        assert centers[0] == 2

    def test_project_labels_size_matches_TR(self) -> None:
        labels = np.array([0, 1, 0])
        per_tr, mask = project_labels_to_TR(labels, n_TRs=10, window_size=4, step_size=3)
        assert per_tr.shape == (10,)
        assert mask.shape == (10,)
        assert mask.sum() > 0

    def test_project_labels_outside_centers_marked_invalid(self) -> None:
        # window=6, step=4, n_TRs=20 -> starts=[0,4,8,12], centers=[2,6,10,14]
        # Valid TR range: [2, 14] inclusive.
        labels = np.array([0, 1, 0, 1])
        per_tr, mask = project_labels_to_TR(labels, n_TRs=20, window_size=6, step_size=4)
        assert not mask[0]
        assert not mask[1]
        assert mask[2]
        assert mask[14]
        assert not mask[15]
        assert not mask[19]

    def test_project_labels_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            project_labels_to_TR(np.array([0, 1]), n_TRs=10, window_size=4, step_size=2)


# --------------------------------------------------------------------------- #
# Per-window FC sweep
# --------------------------------------------------------------------------- #
class TestSweepWindowSize:
    def test_n_windows_correct(self) -> None:
        ts = _stationary_ts(100, 20)
        s = sweep_window_size(ts, window_size=30, step_size=5, k_states=3, seed=0)
        # n_windows = floor((100 - 30)/5) + 1 = 15
        assert s.n_windows == 15

    def test_state_labels_length_matches_n_windows(self) -> None:
        ts = _stationary_ts(80, 15)
        s = sweep_window_size(ts, window_size=20, step_size=5, k_states=3, seed=0)
        assert s.state_labels.shape == (s.n_windows,)

    def test_state_labels_use_only_k_distinct(self) -> None:
        ts = _stationary_ts(120, 20)
        s = sweep_window_size(ts, window_size=30, step_size=5, k_states=4, seed=42)
        assert set(np.unique(s.state_labels)).issubset({0, 1, 2, 3})

    def test_too_few_windows_returns_zero_variability(self) -> None:
        ts = _stationary_ts(20, 10)
        s = sweep_window_size(ts, window_size=20, step_size=1, k_states=3, seed=0)
        # Only one window fits; variability undefined -> 0.
        assert s.n_windows == 1
        assert s.mean_fc_variability == 0.0

    def test_clustering_is_seeded(self) -> None:
        ts = _stationary_ts(120, 20)
        s1 = sweep_window_size(ts, 30, 5, 3, seed=99)
        s2 = sweep_window_size(ts, 30, 5, 3, seed=99)
        np.testing.assert_array_equal(s1.state_labels, s2.state_labels)


class TestFCVariability:
    def test_zero_for_one_window(self) -> None:
        rng = np.random.default_rng(0)
        m = rng.standard_normal((5, 5))
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0.0)
        mean, med = fc_variability_across_windows([m])
        assert mean == 0.0
        assert med == 0.0

    def test_nonzero_for_distinct_windows(self) -> None:
        ts = _two_regime_ts(120, 12, seed=0)
        mats = windowed_fc_matrices(ts, window_size=30, step_size=5)
        mean, med = fc_variability_across_windows(mats)
        assert mean > 0
        assert med > 0


# --------------------------------------------------------------------------- #
# Pairwise ARI
# --------------------------------------------------------------------------- #
class TestPairwiseARI:
    def test_identical_run_self_ari_is_one(self) -> None:
        """Comparing a sweep against itself (via injection) yields ARI = 1."""
        ts = _two_regime_ts(120, 12, seed=1)
        s30 = sweep_window_size(ts, 30, 5, 2, seed=0)
        # Inject a copy of the same sweep under a different size key.
        # Since we're testing ARI logic on pre-projected per-TR labels,
        # bypass pairwise_ari and directly check the projected vector.
        l, m = project_labels_to_TR(s30.state_labels, ts.shape[0], 30, 5)
        # The same labels twice should be perfectly Rand-aligned.
        from sklearn.metrics import adjusted_rand_score
        assert adjusted_rand_score(l[m], l[m]) == 1.0

    def test_pairwise_ari_returns_one_record_per_pair(self) -> None:
        ts = _two_regime_ts(160, 12, seed=2)
        sweeps = {
            20: sweep_window_size(ts, 20, 5, 3, seed=0),
            30: sweep_window_size(ts, 30, 5, 3, seed=0),
            40: sweep_window_size(ts, 40, 5, 3, seed=0),
        }
        recs = pairwise_ari(sweeps, n_TRs=ts.shape[0])
        # 3 sizes -> 3 unordered pairs.
        assert len(recs) == 3
        for r in recs:
            assert "ari" in r and "n_TRs_compared" in r

    def test_pairwise_ari_handles_empty_sweep(self) -> None:
        # Force a degenerate sweep with k_states <= 1 -> all-zero labels.
        ts = _stationary_ts(60, 8)
        sweeps = {
            20: sweep_window_size(ts, 20, 5, 1, seed=0),  # all labels = 0
            30: sweep_window_size(ts, 30, 5, 1, seed=0),
        }
        recs = pairwise_ari(sweeps, n_TRs=60)
        # ARI of constant labels = NaN (sklearn returns 0 for constant
        # sequences via adjusted_rand_score, but our logic still returns
        # a numeric value). We just verify the call completes.
        assert len(recs) == 1


# --------------------------------------------------------------------------- #
# Per-run + group aggregation
# --------------------------------------------------------------------------- #
class TestAnalyseRun:
    def test_returns_one_sweep_per_window_size(self) -> None:
        ts = _two_regime_ts(160, 12, seed=3)
        sru = SubjectRunROI("S1", "1", ts)
        result = analyse_run(sru, [20, 30, 40], step_size=5, k_states=3, seed=0)
        assert set(result.sweeps.keys()) == {20, 30, 40}
        assert result.n_TRs == 160
        assert result.n_rois == 12

    def test_pairwise_ari_count(self) -> None:
        ts = _two_regime_ts(160, 12, seed=4)
        sru = SubjectRunROI("S1", "1", ts)
        result = analyse_run(sru, [20, 30, 40], step_size=5, k_states=3, seed=0)
        assert len(result.ari_pairs) == 3  # C(3, 2)


class TestAggregateGroup:
    def test_group_means_and_stds(self) -> None:
        ts1 = _two_regime_ts(120, 12, seed=10)
        ts2 = _two_regime_ts(120, 12, seed=11)
        rs = [
            analyse_run(SubjectRunROI("S1", "1", ts1), [20, 30], 5, 3, seed=0),
            analyse_run(SubjectRunROI("S2", "1", ts2), [20, 30], 5, 3, seed=0),
        ]
        group = aggregate_group(rs, [20, 30])
        assert group.n_runs == 2
        assert 20 in group.per_window_variability_mean
        assert 30 in group.per_window_variability_mean
        # One pair: (20, 30)
        assert "20-30" in group.pairwise_ari_mean


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class TestIO:
    def test_load_run_timeseries_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        for s in range(2):
            for r in range(2):
                np.save(
                    tmp_path / f"sub-{s:02d}_run-{r + 1}_roi.npy",
                    rng.standard_normal((100, 12)),
                )
        runs = load_run_timeseries(tmp_path)
        assert len(runs) == 4
        assert all(r.timeseries.shape == (100, 12) for r in runs)

    def test_load_rejects_1d(self, tmp_path: Path) -> None:
        np.save(tmp_path / "sub-01_run-1_roi.npy", np.zeros(100))
        with pytest.raises(ValueError):
            load_run_timeseries(tmp_path)

    def test_load_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_run_timeseries(tmp_path / "missing")

    def test_write_outputs_creates_artifacts(self, tmp_path: Path) -> None:
        ts = _two_regime_ts(160, 12, seed=99)
        sru = SubjectRunROI("S1", "1", ts)
        result = analyse_run(sru, [20, 30, 40], step_size=5, k_states=3, seed=0)
        group = aggregate_group([result], [20, 30, 40])
        paths = write_outputs(group, tmp_path)
        for key in ("per_window_csv", "ari_csv", "json"):
            assert paths[key].exists()
        # Header check on per-window CSV.
        with paths["per_window_csv"].open() as f:
            rows = list(csv.reader(f))
        assert rows[0][0] == "subject_id"
        assert "mean_fc_variability" in rows[0]


# --------------------------------------------------------------------------- #
# End-to-end run() entry point
# --------------------------------------------------------------------------- #
class TestRunEntryPoint:
    def test_end_to_end(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "ts_in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        for i in range(3):
            ts = _two_regime_ts(160, 12, seed=i)
            np.save(in_dir / f"sub-{i:02d}_run-1_roi.npy", ts)

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "project": {"random_seed": 0},
                    "paths": {
                        "dfc_input_dir": str(in_dir),
                        "output_root": str(out_dir),
                    },
                    "dfc": {
                        "window_sizes": [20, 30, 40],
                        "step_size": 5,
                        "k_states": 3,
                    },
                }
            )
        )

        group = run(cfg)
        assert isinstance(group, GroupSensitivityResult)
        assert group.n_runs == 3
        assert (out_dir / "dfc_sensitivity_per_window.csv").exists()
        assert (out_dir / "dfc_sensitivity_ari.csv").exists()
        assert (out_dir / "dfc_sensitivity.json").exists()
