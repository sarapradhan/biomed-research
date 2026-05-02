"""Synthetic-data tests for network_anchor (JEI revision Phase 3).

The test design builds FC matrices with controlled block structure and
verifies that the analysis recovers the expected verdict (within > between
under a permutation test, positive modularity, correct reordering, etc.).
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

from fmri_pipeline.reproducibility.network_anchor import (  # noqa: E402
    NetworkAnchorResult,
    analyse,
    assign_networks,
    average_fc_matrices,
    block_means,
    load_group_fc_and_labels,
    newman_modularity,
    parse_schaefer_label,
    permutation_test_within_minus_between,
    reorder_by_network,
    run,
    write_outputs,
)


# --------------------------------------------------------------------------- #
# Synthetic FC builders
# --------------------------------------------------------------------------- #
def _block_fc(
    n_per_net: int = 10,
    networks: tuple = ("Vis", "SomMot", "Default"),
    within_fc: float = 0.6,
    between_fc: float = 0.05,
    noise: float = 0.05,
    seed: int = 0,
):
    """Build a synthetic FC matrix with strong within-network blocks."""
    rng = np.random.default_rng(seed)
    n = n_per_net * len(networks)
    fc = np.full((n, n), between_fc, dtype=float)
    for k, _ in enumerate(networks):
        s = k * n_per_net
        e = s + n_per_net
        fc[s:e, s:e] = within_fc
    fc = fc + rng.standard_normal((n, n)) * noise
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)

    # Schaefer-style ROI labels.
    labels = [
        f"7Networks_LH_{net}_{i + 1}"
        for net in networks
        for i in range(n_per_net)
    ]
    return fc, labels


def _null_fc(n: int = 30, seed: int = 1):
    """No block structure: all entries iid Gaussian."""
    rng = np.random.default_rng(seed)
    fc = rng.standard_normal((n, n))
    fc = (fc + fc.T) / 2.0
    np.fill_diagonal(fc, 0.0)
    labels = [
        f"7Networks_LH_Vis_{i + 1}" if i < n // 2 else f"7Networks_LH_Default_{i + 1}"
        for i in range(n)
    ]
    return fc, labels


# --------------------------------------------------------------------------- #
# Schaefer label parser
# --------------------------------------------------------------------------- #
class TestParseSchaeferLabel:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("7Networks_LH_Vis_1", "Vis"),
            ("7Networks_RH_SomMot_3", "SomMot"),
            ("7Networks_LH_DorsAttn_1", "DorsAttn"),
            ("17Networks_RH_DefaultB_PCC_1", "Default"),
            ("17Networks_LH_ContA_PFCl_2", "Cont"),
            ("17Networks_LH_SalVentAttnA_PrC_1", "SalVentAttn"),
        ],
    )
    def test_known_tokens(self, label, expected) -> None:
        assert parse_schaefer_label(label) == expected

    def test_returns_none_for_unknown(self) -> None:
        assert parse_schaefer_label("AAL_Hippocampus_L") is None

    def test_handles_non_string(self) -> None:
        assert parse_schaefer_label(None) is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Network assignment
# --------------------------------------------------------------------------- #
class TestAssignNetworks:
    def test_assigns_for_each_roi(self) -> None:
        labels = ["7Networks_LH_Vis_1", "7Networks_LH_SomMot_2", "AAL_Foo"]
        nets = assign_networks(labels, fallback="Other")
        assert nets == ["Vis", "SomMot", "Other"]

    def test_overrides_take_precedence(self) -> None:
        labels = ["7Networks_LH_Vis_1", "7Networks_LH_Vis_2"]
        nets = assign_networks(labels, overrides={1: "Custom"})
        assert nets == ["Vis", "Custom"]


# --------------------------------------------------------------------------- #
# Block means
# --------------------------------------------------------------------------- #
class TestBlockMeans:
    def test_within_block_mean_close_to_truth(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, between_fc=0.0, noise=0.0)
        nets = assign_networks(labels)
        means, counts = block_means(fc, nets)
        # Each within-block mean should be exactly 0.7.
        for net in {"Vis", "SomMot", "Default"}:
            key = "__".join(sorted([net, net]))
            assert abs(means[key] - 0.7) < 1e-9
        # Edge counts: within = C(10, 2) = 45 per network.
        for net in {"Vis", "SomMot", "Default"}:
            assert counts["__".join(sorted([net, net]))] == 45

    def test_between_block_mean_close_to_truth(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, between_fc=0.05, noise=0.0)
        nets = assign_networks(labels)
        means, _ = block_means(fc, nets)
        for net_a in ("Vis", "SomMot", "Default"):
            for net_b in ("Vis", "SomMot", "Default"):
                if net_a == net_b:
                    continue
                key = "__".join(sorted([net_a, net_b]))
                assert abs(means[key] - 0.05) < 1e-9


# --------------------------------------------------------------------------- #
# Permutation test
# --------------------------------------------------------------------------- #
class TestPermutationTest:
    def test_significant_for_block_structured(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, between_fc=0.05, noise=0.05)
        nets = assign_networks(labels)
        p, null = permutation_test_within_minus_between(
            fc, nets, n_permutations=500, random_state=0
        )
        assert p < 0.01
        assert null.shape == (500,)

    def test_not_significant_for_null(self) -> None:
        fc, labels = _null_fc(n=30, seed=1)
        nets = assign_networks(labels)
        p, _ = permutation_test_within_minus_between(
            fc, nets, n_permutations=500, random_state=0
        )
        assert p > 0.05

    def test_p_is_bounded(self) -> None:
        fc, labels = _block_fc()
        nets = assign_networks(labels)
        p, _ = permutation_test_within_minus_between(
            fc, nets, n_permutations=200, random_state=0
        )
        assert 0.0 < p <= 1.0


# --------------------------------------------------------------------------- #
# Modularity
# --------------------------------------------------------------------------- #
class TestModularity:
    def test_positive_for_block_structured(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, between_fc=0.05, noise=0.05)
        nets = assign_networks(labels)
        q = newman_modularity(fc, nets)
        assert q > 0.1

    def test_near_zero_for_null(self) -> None:
        fc, labels = _null_fc(n=30, seed=2)
        nets = assign_networks(labels)
        q = newman_modularity(fc, nets)
        assert abs(q) < 0.05

    def test_zero_when_all_one_network(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, noise=0.0)
        # Force every node into the same network.
        q = newman_modularity(fc, ["Vis"] * len(labels))
        # When there is only one community, Q = 0 by definition.
        assert abs(q) < 1e-9


# --------------------------------------------------------------------------- #
# Reordering
# --------------------------------------------------------------------------- #
class TestReorderByNetwork:
    def test_canonical_order_used_by_default(self) -> None:
        fc, labels = _block_fc(networks=("Default", "SomMot", "Vis"))
        nets = assign_networks(labels)
        _, _, ordered = reorder_by_network(fc, nets)
        # Vis should come before SomMot, which comes before Default
        # (canonical Schaefer order).
        assert ordered[0] == "Vis"
        assert ordered[10] == "SomMot"
        assert ordered[20] == "Default"

    def test_custom_order_respected(self) -> None:
        fc, labels = _block_fc(networks=("Vis", "SomMot", "Default"))
        nets = assign_networks(labels)
        _, _, ordered = reorder_by_network(
            fc, nets, network_order=["Default", "Vis", "SomMot"]
        )
        assert ordered[0] == "Default"
        assert ordered[10] == "Vis"
        assert ordered[20] == "SomMot"

    def test_reordered_matrix_preserves_within_block_mass(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, between_fc=0.05, noise=0.0)
        nets = assign_networks(labels)
        reordered, _, _ = reorder_by_network(fc, nets)
        # Sum is invariant under symmetric permutation.
        assert abs(reordered.sum() - fc.sum()) < 1e-9


# --------------------------------------------------------------------------- #
# analyse() top-level function
# --------------------------------------------------------------------------- #
class TestAnalyse:
    def test_block_structured_passes_all_checks(self) -> None:
        fc, labels = _block_fc(within_fc=0.7, between_fc=0.05, noise=0.05)
        result = analyse(fc, labels, n_permutations=500, random_state=0)

        assert isinstance(result, NetworkAnchorResult)
        assert result.n_rois == 30
        assert result.n_networks == 3
        assert result.within_mean > result.between_mean
        assert result.gap_mean > 0
        assert result.p_value < 0.01
        assert result.modularity_q > 0.1
        assert result.reordered_matrix.shape == (30, 30)

    def test_raises_on_non_square_fc(self) -> None:
        with pytest.raises(ValueError):
            analyse(np.zeros((5, 6)), ["a"] * 5)

    def test_raises_on_label_count_mismatch(self) -> None:
        with pytest.raises(ValueError):
            analyse(np.zeros((5, 5)), ["a"] * 4)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
class TestIO:
    def test_load_group_fc_and_labels_csv(self, tmp_path: Path) -> None:
        fc, labels = _block_fc()
        fc_path = tmp_path / "fc.npy"
        np.save(fc_path, fc)
        labels_path = tmp_path / "labels.csv"
        with labels_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label"])
            for lbl in labels:
                w.writerow([lbl])
        loaded_fc, loaded_labels = load_group_fc_and_labels(fc_path, labels_path)
        np.testing.assert_allclose(loaded_fc, fc)
        assert loaded_labels == labels

    def test_load_group_fc_and_labels_json(self, tmp_path: Path) -> None:
        fc, labels = _block_fc()
        fc_path = tmp_path / "fc.npy"
        np.save(fc_path, fc)
        labels_path = tmp_path / "labels.json"
        labels_path.write_text("[" + ", ".join(f'"{l}"' for l in labels) + "]")
        _, loaded_labels = load_group_fc_and_labels(fc_path, labels_path)
        assert loaded_labels == labels

    def test_load_raises_on_label_count_mismatch(self, tmp_path: Path) -> None:
        np.save(tmp_path / "fc.npy", np.zeros((5, 5)))
        (tmp_path / "labels.csv").write_text("a\nb\nc\n")
        with pytest.raises(ValueError):
            load_group_fc_and_labels(tmp_path / "fc.npy", tmp_path / "labels.csv")

    def test_average_fc_matrices_equals_mean(self, tmp_path: Path) -> None:
        fc1, _ = _block_fc(seed=0)
        fc2, _ = _block_fc(seed=1)
        np.save(tmp_path / "sub-01_run-1_fc.npy", fc1)
        np.save(tmp_path / "sub-02_run-1_fc.npy", fc2)
        avg = average_fc_matrices(tmp_path)
        np.testing.assert_allclose(avg, (fc1 + fc2) / 2.0)

    def test_average_fc_matrices_raises_on_empty(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            average_fc_matrices(tmp_path)

    def test_write_outputs_creates_artifacts(self, tmp_path: Path) -> None:
        fc, labels = _block_fc()
        result = analyse(fc, labels, n_permutations=100, random_state=0)
        paths = write_outputs(result, tmp_path)
        for key in ("summary", "blocks", "matrix", "label_order", "json"):
            assert paths[key].exists(), f"missing {key}"

        # summary CSV should not contain array fields
        with paths["summary"].open() as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["metric", "value"]
        keys = {r[0] for r in rows[1:]}
        for k in ("reordered_matrix", "block_means", "block_counts", "network_assignment"):
            assert k not in keys


# --------------------------------------------------------------------------- #
# End-to-end run() entry point
# --------------------------------------------------------------------------- #
class TestRunEntryPoint:
    def test_end_to_end_with_per_run_fc(self, tmp_path: Path) -> None:
        in_dir = tmp_path / "fc_in"
        out_dir = tmp_path / "out"
        in_dir.mkdir()
        # Write 3 per-run FC matrices that all share the block structure.
        for i in range(3):
            fc, labels = _block_fc(seed=i, noise=0.05)
            np.save(in_dir / f"sub-{i:02d}_run-1_fc.npy", fc)
        labels_path = tmp_path / "labels.csv"
        with labels_path.open("w", newline="") as f:
            w = csv.writer(f)
            for lbl in labels:
                w.writerow([lbl])

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "project": {"random_seed": 0},
                    "paths": {
                        "fc_input_dir": str(in_dir),
                        "roi_labels": str(labels_path),
                        "output_root": str(out_dir),
                    },
                    "network_anchor": {"n_permutations": 200},
                }
            )
        )

        result = run(cfg)
        assert result.gap_mean > 0
        assert result.p_value < 0.05
        assert result.modularity_q > 0.05
        for fname in (
            "network_anchor_summary.csv",
            "network_anchor_block_means.csv",
            "network_anchor_reordered.npy",
            "network_anchor_label_order.csv",
            "network_anchor.json",
        ):
            assert (out_dir / fname).exists()
