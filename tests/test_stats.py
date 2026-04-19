from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.stats import (
    build_design_matrix,
    edge_results_table,
    mass_univariate_ols,
    network_summary,
)


class TestMassUnivariateOLS:
    """Tests for mass_univariate_ols function."""

    def test_mass_univariate_ols_shape(self) -> None:
        """Test that OLS output has expected shapes."""
        y = np.random.randn(20, 50)
        x = np.column_stack([
            np.ones(20),
            np.random.binomial(1, 0.5, 20),  # diagnosis
            np.random.randn(20),  # covariate
        ])
        result = mass_univariate_ols(y, x, contrast_idx=1)

        assert result["beta"].shape == (50,)
        assert result["t"].shape == (50,)
        assert result["p"].shape == (50,)
        assert result["q"].shape == (50,)
        assert result["dof"].shape == (1,)

    def test_mass_univariate_ols_perfect_signal(self) -> None:
        """Test with known perfect signal."""
        n_subjects = 30
        n_features = 20

        # Create y where feature 0 is perfectly predicted by diagnosis
        diagnosis = np.random.binomial(1, 0.5, n_subjects)
        y = np.zeros((n_subjects, n_features))
        y[:, 0] = diagnosis.astype(float) * 10.0  # Perfect correlation

        # Random features
        y[:, 1:] = np.random.randn(n_subjects, n_features - 1)

        x = np.column_stack([
            np.ones(n_subjects),
            diagnosis.astype(float),
        ])

        result = mass_univariate_ols(y, x, contrast_idx=1)

        # Feature 0 should have high t-value
        assert np.abs(result["t"][0]) > 5.0
        # Feature 0 should have very low p-value
        assert result["p"][0] < 0.001

    def test_mass_univariate_ols_dof_calculation(self) -> None:
        """Test degrees of freedom calculation."""
        n_subjects = 50
        n_features = 10
        n_regressors = 3

        y = np.random.randn(n_subjects, n_features)
        x = np.random.randn(n_subjects, n_regressors)

        result = mass_univariate_ols(y, x, contrast_idx=1)

        expected_dof = n_subjects - n_regressors
        assert result["dof"][0] == expected_dof

    def test_mass_univariate_ols_q_values(self) -> None:
        """Test that q-values are FDR-corrected."""
        y = np.random.randn(20, 50)
        x = np.column_stack([
            np.ones(20),
            np.random.binomial(1, 0.5, 20),
        ])
        result = mass_univariate_ols(y, x, contrast_idx=1)

        # q-values should be >= p-values (FDR correction)
        assert np.all(result["q"] >= result["p"])

    def test_mass_univariate_ols_univariate_property(self) -> None:
        """Test independence of features (univariate property)."""
        n_subjects = 20
        n_features = 10

        # Create y where only feature 0 has signal
        diagnosis = np.random.binomial(1, 0.5, n_subjects)
        y = np.zeros((n_subjects, n_features))
        y[:, 0] = diagnosis.astype(float) * 5.0  # Signal in feature 0
        y[:, 1:] = np.random.randn(n_subjects, n_features - 1)  # Noise elsewhere

        x = np.column_stack([np.ones(n_subjects), diagnosis.astype(float)])

        result = mass_univariate_ols(y, x, contrast_idx=1)

        # Feature 0 should have smallest (most significant) p-value
        assert np.argmin(result["p"]) == 0


class TestEdgeResultsTable:
    """Tests for edge_results_table function."""

    def test_edge_results_table_columns(self) -> None:
        """Test that output DataFrame has expected columns."""
        beta = np.array([0.1, 0.2, 0.3])
        tvals = np.array([2.0, 3.0, 1.5])
        pvals = np.array([0.05, 0.01, 0.1])
        qvals = np.array([0.05, 0.03, 0.1])
        roi_i = np.array([0, 0, 1])
        roi_j = np.array([1, 2, 2])
        edge_idx = (roi_i, roi_j)
        roi_labels = ["ROI_0", "ROI_1", "ROI_2"]

        df = edge_results_table(beta, tvals, pvals, qvals, edge_idx, roi_labels, "static_fc")

        expected_cols = ["metric", "roi_i", "roi_j", "label_i", "label_j",
                        "beta_diagnosis", "t", "p", "q", "significant_fdr"]
        for col in expected_cols:
            assert col in df.columns

    def test_edge_results_table_sorting(self) -> None:
        """Test that results are sorted by q-value."""
        beta = np.array([0.1, 0.2, 0.3])
        tvals = np.array([2.0, 3.0, 1.5])
        pvals = np.array([0.05, 0.01, 0.1])
        qvals = np.array([0.1, 0.03, 0.05])  # Not sorted
        roi_i = np.array([0, 0, 1])
        roi_j = np.array([1, 2, 2])
        edge_idx = (roi_i, roi_j)
        roi_labels = ["ROI_0", "ROI_1", "ROI_2"]

        df = edge_results_table(beta, tvals, pvals, qvals, edge_idx, roi_labels, "static_fc")

        # Should be sorted by q
        assert list(df["q"]) == [0.03, 0.05, 0.1]

    def test_edge_results_table_significant_flag(self) -> None:
        """Test significant_fdr flag (q < 0.05)."""
        beta = np.array([0.1, 0.2, 0.3])
        tvals = np.array([2.0, 3.0, 1.5])
        pvals = np.array([0.05, 0.01, 0.1])
        qvals = np.array([0.06, 0.03, 0.1])
        roi_i = np.array([0, 0, 1])
        roi_j = np.array([1, 2, 2])
        edge_idx = (roi_i, roi_j)
        roi_labels = ["ROI_0", "ROI_1", "ROI_2"]

        df = edge_results_table(beta, tvals, pvals, qvals, edge_idx, roi_labels, "static_fc")

        # Table sorted by q asc: [0.03, 0.06, 0.1]
        assert df["significant_fdr"].iloc[0] == True   # q=0.03 (smallest, significant)
        assert df["significant_fdr"].iloc[1] == False  # q=0.06
        assert df["significant_fdr"].iloc[2] == False  # q=0.1

    def test_edge_results_table_roi_labels(self) -> None:
        """Test that ROI labels are correctly mapped."""
        beta = np.array([0.1])
        tvals = np.array([2.0])
        pvals = np.array([0.05])
        qvals = np.array([0.05])
        roi_i = np.array([1])
        roi_j = np.array([2])
        edge_idx = (roi_i, roi_j)
        roi_labels = ["ROI_0", "ROI_1", "ROI_2", "ROI_3"]

        df = edge_results_table(beta, tvals, pvals, qvals, edge_idx, roi_labels, "static_fc")

        assert df["label_i"].iloc[0] == "ROI_1"
        assert df["label_j"].iloc[0] == "ROI_2"


class TestNetworkSummary:
    """Tests for network_summary function."""

    def test_network_summary_empty(self) -> None:
        """Test empty input produces empty summary."""
        df = pd.DataFrame({
            "significant_fdr": [False, False],
            "label_i": ["ROI_0", "ROI_1"],
            "label_j": ["ROI_1", "ROI_2"],
            "beta_diagnosis": [0.1, 0.2],
        })

        result = network_summary(df)

        assert result.empty
        assert list(result.columns) == ["network_pair", "n_edges", "mean_beta"]

    def test_network_summary_basic(self) -> None:
        """Test basic network summary."""
        df = pd.DataFrame({
            "significant_fdr": [True, True, False],
            "label_i": ["7Networks_DMN_1", "7Networks_DMN_2", "7Networks_VAN_1"],
            "label_j": ["7Networks_DMN_3", "7Networks_DMN_4", "7Networks_VAN_2"],
            "beta_diagnosis": [0.5, 0.3, 0.2],
        })

        result = network_summary(df)

        assert len(result) == 1
        assert result["n_edges"].iloc[0] == 2
        assert np.isclose(result["mean_beta"].iloc[0], (0.5 + 0.3) / 2)

    def test_network_summary_cross_network(self) -> None:
        """Test summary with cross-network edges."""
        df = pd.DataFrame({
            "significant_fdr": [True, True, True],
            "label_i": ["7Networks_DMN_1", "7Networks_DMN_2", "7Networks_VAN_1"],
            "label_j": ["7Networks_DMN_3", "7Networks_VAN_2", "7Networks_VAN_3"],
            "beta_diagnosis": [0.5, 0.3, 0.2],
        })

        result = network_summary(df)

        # DMN-DMN: 1 edge
        # DMN-VAN: 1 edge
        # VAN-VAN: 1 edge
        assert len(result) == 3

    def test_network_summary_ordering(self) -> None:
        """Test that network pairs are consistently ordered."""
        df = pd.DataFrame({
            "significant_fdr": [True, True],
            "label_i": ["7Networks_A_1", "7Networks_B_1"],
            "label_j": ["7Networks_B_2", "7Networks_A_2"],
            "beta_diagnosis": [0.5, 0.3],
        })

        result = network_summary(df)

        # Both should result in A__B pairs
        assert "A__B" in result["network_pair"].values

    def test_network_summary_grouped_correctly(self) -> None:
        """Test grouping by network pair."""
        df = pd.DataFrame({
            "significant_fdr": [True, True, True, True],
            "label_i": ["7Networks_X_1", "7Networks_X_2", "7Networks_X_3", "7Networks_Y_1"],
            "label_j": ["7Networks_X_2", "7Networks_X_3", "7Networks_X_4", "7Networks_Y_2"],
            "beta_diagnosis": [0.1, 0.2, 0.3, 0.4],
        })

        result = network_summary(df)

        # Should have X__X: 3 edges and X__Y: 1 edge
        xx = result[result["network_pair"] == "X__X"]
        assert len(xx) == 1
        assert xx["n_edges"].iloc[0] == 3
        assert np.isclose(xx["mean_beta"].iloc[0], (0.1 + 0.2 + 0.3) / 3)


class TestBuildDesignMatrix:
    """Tests for build_design_matrix function."""

    def test_build_design_matrix_basic(self) -> None:
        """Test basic design matrix building."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": ["age"],
                "optional_covariates_if_available": [],
            }
        }
        df = pd.DataFrame({
            "subject": ["01", "02", "03", "04"],
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
            "age": [30.0, 28.0, 33.0, 29.0],
        })

        x, cols, out = build_design_matrix(df, cfg)

        assert x.shape[0] == 4
        assert x.shape[1] == 3  # intercept, diagnosis, age
        assert cols[0] == "intercept"
        assert cols[1] == "diagnosis_bin"
        assert "age" in cols

    def test_build_design_matrix_intercept(self) -> None:
        """Test that intercept column is all ones."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": [],
                "optional_covariates_if_available": [],
            }
        }
        df = pd.DataFrame({
            "subject": ["01", "02", "03"],
            "diagnosis": ["SZ", "HC", "SZ"],
        })

        x, _, _ = build_design_matrix(df, cfg)

        assert np.allclose(x[:, 0], 1.0)

    def test_build_design_matrix_diagnosis_binary(self) -> None:
        """Test that diagnosis is correctly binarized."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": [],
                "optional_covariates_if_available": [],
            }
        }
        df = pd.DataFrame({
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
        })

        x, cols, _ = build_design_matrix(df, cfg)

        assert cols[1] == "diagnosis_bin"
        assert np.array_equal(x[:, 1], [1, 0, 1, 0])

    def test_build_design_matrix_covariate_standardization(self) -> None:
        """Test that continuous covariates are standardized."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": ["age"],
                "optional_covariates_if_available": [],
            }
        }
        df = pd.DataFrame({
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
            "age": [30.0, 28.0, 33.0, 29.0],
        })

        x, cols, _ = build_design_matrix(df, cfg)

        age_col = x[:, cols.index("age")]
        # Should be standardized: mean ~0, std ~1
        assert np.isclose(np.mean(age_col), 0.0, atol=1e-10)
        assert np.isclose(np.std(age_col), 1.0, atol=1e-10)

    def test_build_design_matrix_optional_covariates(self) -> None:
        """Test optional covariate inclusion."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": [],
                "optional_covariates_if_available": ["sex", "site"],
            }
        }
        df = pd.DataFrame({
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
            "sex": ["M", "F", "M", "F"],
            # site is intentionally missing
        })

        x, cols, _ = build_design_matrix(df, cfg)

        # sex should be included (categorical with dummy coding)
        assert any("sex" in c for c in cols)
        # site should not appear in columns
        assert not any("site" in c for c in cols)

    def test_build_design_matrix_categorical_coding(self) -> None:
        """Test categorical covariate dummy coding."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": ["site"],
                "optional_covariates_if_available": [],
            }
        }
        df = pd.DataFrame({
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
            "site": ["A", "A", "B", "B"],
        })

        x, cols, _ = build_design_matrix(df, cfg)

        # Should have intercept, diagnosis, and site dummy (drop_first=True)
        # Only one level should be included
        assert any("site" in c for c in cols)

    def test_build_design_matrix_missing_values(self) -> None:
        """Test handling of missing covariate values."""
        cfg = {
            "stats": {
                "diagnosis_column": "diagnosis",
                "patient_label": "SZ",
                "control_label": "HC",
                "covariates": ["age"],
                "optional_covariates_if_available": [],
            }
        }
        df = pd.DataFrame({
            "diagnosis": ["SZ", "HC", "SZ", "HC"],
            "age": [30.0, np.nan, 33.0, 29.0],
        })

        x, cols, _ = build_design_matrix(df, cfg)

        # Missing values should be imputed with mean
        age_col = x[:, cols.index("age")]
        assert not np.any(np.isnan(age_col))
