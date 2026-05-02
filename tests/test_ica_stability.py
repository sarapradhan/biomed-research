"""Synthetic-data tests for fmri_pipeline.reproducibility.ica_stability.

Strategy: generate a fixed set of "ground-truth" spatial component
templates, then simulate "decompositions" by adding small noise (and
randomly permuting + sign-flipping) to those templates. Hungarian
matching should recover the underlying alignment, and matched
correlations should be high. The null case uses independent random
components, where matched |r| should be small.
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

from fmri_pipeline.reproducibility.ica_stability import (  # noqa: E402
    ICARun,
    ICAStabilityResult,
    load_ica_runs,
    match_components_hungarian,
    run,
    stability_across_run_subsets,
    stability_across_seeds,
    write_stability_outputs,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_perturbed_run(
    templates: np.ndarray,
    rng: np.random.Generator,
    noise: float = 0.1,
    permute: bool = True,
    flip_signs: bool = True,
) -> np.ndarray:
    """Return a perturbed copy of the K-template matrix.

    Models the same decomposition recovered with a different seed:
    components are in a different order, may flip sign, and have a small
    additive noise.
    """
    K, F = templates.shape
    out = templates + rng.standard_normal((K, F)) * noise
    if flip_signs:
        signs = rng.choice([-1.0, 1.0], size=(K, 1))
        out = out * signs
    if permute:
        perm = rng.permutation(K)
        out = out[perm]
    return out


def _make_consistent_runs(
    n_runs: int = 5,
    K: int = 6,
    F: int = 200,
    noise: float = 0.05,
    seed: int = 42,
):
    """High-stability scenario: every decomposition is a perturbation of one template set."""
    rng = np.random.default_rng(seed)
    templates = rng.standard_normal((K, F))
    runs = []
    for i in range(n_runs):
        comps = _make_perturbed_run(templates, rng, noise=noise)
        runs.append(ICARun(tag=f"seed-{i + 1}", components=comps))
    return templates, runs


def _make_null_runs(n_runs: int = 5, K: int = 6, F: int = 200, seed: int = 7):
    """Low-stability scenario: every decomposition is independent random."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(n_runs):
        comps = rng.standard_normal((K, F))
        runs.append(ICARun(tag=f"seed-{i + 1}", components=comps))
    return runs


# --------------------------------------------------------------------------- #
# Hungarian matching
# --------------------------------------------------------------------------- #
class TestMatchComponentsHungarian:
    def test_matches_identical_components(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 50))
        # Exact copy with rows permuted -> matched_r should all be ~1.0.
        perm = rng.permutation(4)
        b = a[perm]
        row_ind, col_ind, matched_r = match_components_hungarian(a, b)
        assert row_ind.shape == (4,)
        assert col_ind.shape == (4,)
        assert np.all(matched_r > 0.999)
        # Recovered permutation should be exactly perm.
        # row_ind[k] = k for some k (linear_sum_assignment returns sorted rows
        # for square cost matrices), col_ind tells us which b row matched.
        np.testing.assert_array_equal(col_ind[np.argsort(row_ind)], perm)

    def test_handles_sign_flipped_components(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((4, 50))
        b = -a  # exact sign flip
        _, _, matched_r = match_components_hungarian(a, b)
        # |r| ignores sign: matched correlations must be ~1.0.
        assert np.all(matched_r > 0.999)

    def test_independent_components_have_low_match(self) -> None:
        rng = np.random.default_rng(2)
        a = rng.standard_normal((6, 200))
        b = rng.standard_normal((6, 200))
        _, _, matched_r = match_components_hungarian(a, b)
        # Even after Hungarian, mean |r| of independent gaussians is small.
        assert matched_r.mean() < 0.4

    def test_raises_on_feature_mismatch(self) -> None:
        a = np.zeros((3, 50))
        b = np.zeros((3, 30))
        with pytest.raises(ValueError):
            match_components_hungarian(a, b)


# --------------------------------------------------------------------------- #
# Stability across seeds
# --------------------------------------------------------------------------- #
class TestStabilityAcrossSeeds:
    def test_consistent_runs_yield_high_robustness(self) -> None:
        _, runs = _make_consistent_runs(n_runs=5, K=6, noise=0.05)
        result = stability_across_seeds(runs, robust_threshold=0.7)

        assert isinstance(result, ICAStabilityResult)
        assert result.sweep == "seeds"
        assert result.k_components == 6
        assert result.n_runs == 5
        # 5 choose 2 = 10 unordered pairs.
        assert result.pairwise_runs_considered == 10
        # With low noise + permutation, all components should be recovered.
        assert result.n_robust_components == 6
        assert result.mean_matched_correlation > 0.9

    def test_null_runs_yield_few_robust(self) -> None:
        runs = _make_null_runs(n_runs=5, K=6)
        result = stability_across_seeds(runs, robust_threshold=0.7)
        assert result.n_robust_components == 0
        assert result.mean_matched_correlation < 0.5

    def test_per_component_arrays_match_K(self) -> None:
        _, runs = _make_consistent_runs(n_runs=4, K=5)
        result = stability_across_seeds(runs)
        assert result.per_component_mean_r.shape == (5,)
        assert result.per_component_std_r.shape == (5,)

    def test_pairwise_records_have_expected_keys(self) -> None:
        _, runs = _make_consistent_runs(n_runs=3, K=4)
        result = stability_across_seeds(runs)
        assert len(result.pairwise_records) == 3  # C(3, 2)
        keys = set(result.pairwise_records[0].keys())
        for required in {
            "tag_a",
            "tag_b",
            "k_matched",
            "mean_matched_r",
            "n_above_threshold",
        }:
            assert required in keys

    def test_raises_on_lt_two_runs(self) -> None:
        _, runs = _make_consistent_runs(n_runs=1)
        with pytest.raises(ValueError):
            stability_across_seeds(runs)

    def test_raises_on_K_mismatch(self) -> None:
        rng = np.random.default_rng(0)
        runs = [
            ICARun("a", rng.standard_normal((4, 50))),
            ICARun("b", rng.standard_normal((5, 50))),
        ]
        with pytest.raises(ValueError):
            stability_across_seeds(runs)


# --------------------------------------------------------------------------- #
# Run-subset variant should reach the same code path
# --------------------------------------------------------------------------- #
class TestStabilityAcrossRunSubsets:
    def test_uses_run_subsets_label(self) -> None:
        _, runs = _make_consistent_runs(n_runs=4)
        result = stability_across_run_subsets(runs)
        assert result.sweep == "run_subsets"

    def test_consistency_with_seed_branch(self) -> None:
        _, runs = _make_consistent_runs(n_runs=4, K=5, noise=0.05)
        seeds = stability_across_seeds(runs)
        loro = stability_across_run_subsets(runs)
        # Same numerical aggregation, just different label.
        assert seeds.k_components == loro.k_components
        np.testing.assert_allclose(
            seeds.per_component_mean_r, loro.per_component_mean_r
        )


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class TestIO:
    def test_load_ica_runs_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(0)
        for i in range(3):
            np.save(tmp_path / f"seed-{i + 1}_ica_components.npy", rng.standard_normal((5, 40)))
        runs = load_ica_runs(tmp_path)
        assert len(runs) == 3
        assert all(r.components.shape == (5, 40) for r in runs)
        assert {r.tag for r in runs} == {"seed-1", "seed-2", "seed-3"}

    def test_load_rejects_1d_arrays(self, tmp_path: Path) -> None:
        np.save(tmp_path / "seed-1_ica_components.npy", np.zeros(50))
        with pytest.raises(ValueError):
            load_ica_runs(tmp_path)

    def test_load_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_ica_runs(tmp_path / "missing")

    def test_write_outputs_creates_all_artifacts(self, tmp_path: Path) -> None:
        _, runs = _make_consistent_runs(n_runs=4, K=5)
        result = stability_across_seeds(runs)
        paths = write_stability_outputs(result, tmp_path, summary_csv="ica_test.csv")

        for key in ("csv", "per_component_csv", "pairwise_csv", "json"):
            assert paths[key].exists(), f"Missing artifact: {key}"

        # Summary CSV must not contain array fields as rows.
        with paths["csv"].open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["metric", "value"]
        keys = {r[0] for r in rows[1:]}
        assert "per_component_mean_r" not in keys
        assert "pairwise_records" not in keys

        # Per-component CSV must have one row per component.
        with paths["per_component_csv"].open() as f:
            comp_rows = list(csv.reader(f))
        assert comp_rows[0] == [
            "component_index",
            "mean_matched_r",
            "std_matched_r",
            "robust",
        ]
        assert len(comp_rows) - 1 == result.k_components


# --------------------------------------------------------------------------- #
# End-to-end run() entry point
# --------------------------------------------------------------------------- #
class TestRunEntryPoint:
    def test_run_with_seeds_only(self, tmp_path: Path) -> None:
        seeds_dir = tmp_path / "ica_in" / "seeds"
        seeds_dir.mkdir(parents=True)
        _, runs = _make_consistent_runs(n_runs=4, K=5, noise=0.05)
        for r in runs:
            np.save(seeds_dir / f"{r.tag}_ica_components.npy", r.components)

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "project": {"random_seed": 0},
                    "paths": {
                        "ica_input_dir": str(tmp_path / "ica_in"),
                        "output_root": str(tmp_path / "reports"),
                    },
                    "ica": {
                        "stability": {
                            "robust_threshold": 0.7,
                            "output_seeds_csv": "ica_seeds.csv",
                        }
                    },
                }
            )
        )

        results = run(cfg_path)

        assert "seeds" in results
        assert "run_subsets" not in results
        assert (tmp_path / "reports" / "ica_seeds.csv").exists()
        assert (tmp_path / "reports" / "ica_seeds_per_component.csv").exists()
        assert (tmp_path / "reports" / "ica_seeds_pairwise.csv").exists()
        assert results["seeds"].n_robust_components == 5

    def test_run_raises_when_neither_subdir_present(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "paths": {
                        "ica_input_dir": str(tmp_path / "missing"),
                        "output_root": str(tmp_path / "reports"),
                    },
                    "ica": {"stability": {}},
                }
            )
        )
        with pytest.raises(FileNotFoundError):
            run(cfg_path)
