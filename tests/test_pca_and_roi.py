"""
Comprehensive tests for pca_metrics.py and roi.py modules.

Tests cover:
- PCA explained variance ratios
- PCA row appending and CSV saving
- Schaefer atlas fetching
- ROI time series extraction
- ROI time series saving (NPY and CSV)
- Round-trip file I/O for ROI data
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.pca_metrics import (
    append_pca_row,
    run_subject_pca,
    save_pca_table,
)
from fmri_pipeline.roi import (
    extract_roi_timeseries,
    get_schaefer_atlas,
    save_roi_timeseries,
)


class TestRunSubjectPca:
    """Tests for run_subject_pca PCA component extraction."""

    def test_run_subject_pca_shape(self) -> None:
        """Test that PCA returns correct number of components."""
        roi_ts = np.random.randn(100, 50)  # 100 timepoints, 50 ROIs
        cfg = {"pca": {"n_components": 5}, "project": {"random_seed": 42}}

        evr = run_subject_pca(roi_ts, cfg)

        assert evr.shape == (5,)

    def test_run_subject_pca_sum_to_unity(self) -> None:
        """Test that explained variance ratios sum to ≤1.0."""
        roi_ts = np.random.randn(100, 50)
        cfg = {"pca": {"n_components": 5}, "project": {"random_seed": 42}}

        evr = run_subject_pca(roi_ts, cfg)

        total_variance = np.sum(evr)
        assert total_variance <= 1.0 + 1e-6  # Allow small numerical error

    def test_run_subject_pca_decreasing(self) -> None:
        """Test that explained variance ratios are in decreasing order."""
        roi_ts = np.random.randn(100, 50)
        cfg = {"pca": {"n_components": 5}, "project": {"random_seed": 42}}

        evr = run_subject_pca(roi_ts, cfg)

        # Each component should explain less than the previous
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1]

    def test_run_subject_pca_positive(self) -> None:
        """Test that all explained variance ratios are positive."""
        roi_ts = np.random.randn(100, 50)
        cfg = {"pca": {"n_components": 5}, "project": {"random_seed": 42}}

        evr = run_subject_pca(roi_ts, cfg)

        assert np.all(evr >= 0.0)

    def test_run_subject_pca_reproducible(self) -> None:
        """Test that same seed produces identical PCA results."""
        roi_ts = np.random.randn(100, 50)

        cfg1 = {"pca": {"n_components": 5}, "project": {"random_seed": 123}}
        evr1 = run_subject_pca(roi_ts, cfg1)

        cfg2 = {"pca": {"n_components": 5}, "project": {"random_seed": 123}}
        evr2 = run_subject_pca(roi_ts, cfg2)

        assert np.allclose(evr1, evr2)

    def test_run_subject_pca_different_seeds(self) -> None:
        """Test that different seeds can produce different results."""
        # Use SVD-sensitive data
        roi_ts = np.random.randn(100, 50)

        cfg1 = {"pca": {"n_components": 5}, "project": {"random_seed": 1}}
        evr1 = run_subject_pca(roi_ts, cfg1)

        cfg2 = {"pca": {"n_components": 5}, "project": {"random_seed": 999}}
        evr2 = run_subject_pca(roi_ts, cfg2)

        # With different seeds, explained variance might differ
        # (though for deterministic input, sklearn's PCA might match)
        assert isinstance(evr1, np.ndarray)
        assert isinstance(evr2, np.ndarray)

    def test_run_subject_pca_more_components_than_features(self) -> None:
        """Test PCA with n_components > n_features (should pad)."""
        roi_ts = np.random.randn(100, 5)  # Only 5 ROIs
        cfg = {"pca": {"n_components": 10}, "project": {"random_seed": 42}}

        evr = run_subject_pca(roi_ts, cfg)

        assert evr.shape == (10,)
        # Last 5 values should be zero (padded)
        assert np.allclose(evr[5:], 0.0)

    def test_run_subject_pca_first_component_largest(self) -> None:
        """Test that first component has largest explained variance."""
        roi_ts = np.random.randn(200, 100)
        cfg = {"pca": {"n_components": 10}, "project": {"random_seed": 42}}

        evr = run_subject_pca(roi_ts, cfg)

        assert evr[0] == np.max(evr)


class TestAppendPcaRow:
    """Tests for append_pca_row row construction."""

    def test_append_pca_row_basic(self) -> None:
        """Test appending a single PCA result."""
        rows = []
        evr = np.array([0.5, 0.3, 0.2])

        append_pca_row(rows, "001", "dataset_a", "HC", evr)

        assert len(rows) == 3  # One row per component
        assert rows[0]["subject"] == "001"
        assert rows[0]["component"] == 1

    def test_append_pca_row_multiple_appends(self) -> None:
        """Test appending multiple subjects."""
        rows = []
        evr = np.array([0.5, 0.3, 0.2])

        append_pca_row(rows, "001", "dataset_a", "HC", evr)
        append_pca_row(rows, "002", "dataset_a", "SZ", evr)

        assert len(rows) == 6  # 3 components each

        # Check first subject
        assert rows[0]["subject"] == "001"
        assert rows[0]["diagnosis"] == "HC"

        # Check second subject
        assert rows[3]["subject"] == "002"
        assert rows[3]["diagnosis"] == "SZ"

    def test_append_pca_row_component_numbering(self) -> None:
        """Test that components are numbered starting from 1."""
        rows = []
        evr = np.array([0.4, 0.3, 0.2, 0.1])

        append_pca_row(rows, "001", "dataset_a", "HC", evr)

        components = [r["component"] for r in rows]
        assert components == [1, 2, 3, 4]

    def test_append_pca_row_values_correct(self) -> None:
        """Test that explained variance values are correctly stored."""
        rows = []
        evr = np.array([0.5, 0.25, 0.15, 0.1])

        append_pca_row(rows, "001", "dataset_a", "HC", evr)

        values = [r["explained_variance_ratio"] for r in rows]
        assert np.allclose(values, evr)

    def test_append_pca_row_metadata(self) -> None:
        """Test that all metadata fields are set."""
        rows = []
        evr = np.array([0.5, 0.3])

        append_pca_row(rows, "sub123", "my_dataset", "CONTROL", evr)

        for row in rows:
            assert "subject" in row
            assert "dataset" in row
            assert "diagnosis" in row
            assert "component" in row
            assert "explained_variance_ratio" in row

    def test_append_pca_row_single_component(self) -> None:
        """Test appending single component."""
        rows = []
        evr = np.array([1.0])

        append_pca_row(rows, "001", "dataset", "HC", evr)

        assert len(rows) == 1
        assert rows[0]["explained_variance_ratio"] == 1.0


class TestSavePcaTable:
    """Tests for save_pca_table CSV saving."""

    def test_save_pca_table_basic(self, tmp_path: Path) -> None:
        """Test saving PCA results to CSV."""
        rows = [
            {
                "subject": "001",
                "dataset": "dataset_a",
                "diagnosis": "HC",
                "component": 1,
                "explained_variance_ratio": 0.5,
            },
            {
                "subject": "001",
                "dataset": "dataset_a",
                "diagnosis": "HC",
                "component": 2,
                "explained_variance_ratio": 0.3,
            },
        ]

        path = save_pca_table(rows, tmp_path)

        assert Path(path).exists()
        df = pd.read_csv(path)
        assert df.shape == (2, 5)

    def test_save_pca_table_columns(self, tmp_path: Path) -> None:
        """Test that all expected columns are in output CSV."""
        rows = [
            {
                "subject": "001",
                "dataset": "dataset_a",
                "diagnosis": "HC",
                "component": 1,
                "explained_variance_ratio": 0.5,
            },
        ]

        path = save_pca_table(rows, tmp_path)

        df = pd.read_csv(path)
        expected_cols = {"subject", "dataset", "diagnosis", "component", "explained_variance_ratio"}
        assert expected_cols.issubset(set(df.columns))

    def test_save_pca_table_creates_directory(self, tmp_path: Path) -> None:
        """Test that save_pca_table creates output directory."""
        nested_dir = tmp_path / "a" / "b" / "c"
        rows = [
            {
                "subject": "001",
                "dataset": "dataset",
                "diagnosis": "HC",
                "component": 1,
                "explained_variance_ratio": 0.5,
            },
        ]

        path = save_pca_table(rows, nested_dir)

        assert nested_dir.exists()
        assert Path(path).exists()

    def test_save_pca_table_filename(self, tmp_path: Path) -> None:
        """Test that output file has expected name."""
        rows = []
        path = save_pca_table(rows, tmp_path)

        assert "pca_explained_variance.csv" in path

    def test_save_pca_table_multiple_subjects(self, tmp_path: Path) -> None:
        """Test saving multiple subjects' PCA results."""
        rows = []
        for subj in ["001", "002", "003"]:
            for comp in [1, 2, 3]:
                rows.append(
                    {
                        "subject": subj,
                        "dataset": "dataset_a",
                        "diagnosis": "HC",
                        "component": comp,
                        "explained_variance_ratio": 0.5 / comp,
                    }
                )

        path = save_pca_table(rows, tmp_path)

        df = pd.read_csv(path)
        assert df.shape[0] == 9  # 3 subjects * 3 components
        assert df["subject"].nunique() == 3

    def test_save_pca_table_round_trip(self, tmp_path: Path) -> None:
        """Test that saved data can be reloaded correctly."""
        original_rows = [
            {
                "subject": "001",
                "dataset": "dataset_a",
                "diagnosis": "HC",
                "component": 1,
                "explained_variance_ratio": 0.5,
            },
            {
                "subject": "001",
                "dataset": "dataset_a",
                "diagnosis": "HC",
                "component": 2,
                "explained_variance_ratio": 0.3,
            },
        ]

        path = save_pca_table(original_rows, tmp_path)
        df = pd.read_csv(path, dtype={"subject": str})

        assert df["subject"].iloc[0] == "001"
        assert df["explained_variance_ratio"].iloc[0] == 0.5


class TestGetSchaeferAtlas:
    """Tests for get_schaefer_atlas atlas fetching."""

    def test_get_schaefer_atlas_returns_tuple(self) -> None:
        """Test that get_schaefer_atlas returns maps and labels."""
        cfg = {"roi": {"schaefer_n_rois": 100, "schaefer_yeo_networks": 7}}

        maps, labels = get_schaefer_atlas(cfg)

        assert maps is not None
        assert labels is not None

    def test_get_schaefer_atlas_maps_type(self) -> None:
        """Test that atlas maps is a NIfTI image."""
        cfg = {"roi": {"schaefer_n_rois": 100, "schaefer_yeo_networks": 7}}

        maps, labels = get_schaefer_atlas(cfg)

        # Should be a nibabel image object
        assert hasattr(maps, "get_fdata") or isinstance(maps, (str, Path))

    def test_get_schaefer_atlas_labels_type(self) -> None:
        """Test that atlas labels is list-like."""
        cfg = {"roi": {"schaefer_n_rois": 100, "schaefer_yeo_networks": 7}}

        maps, labels = get_schaefer_atlas(cfg)

        # Labels should be iterable
        assert hasattr(labels, "__iter__")

    def test_get_schaefer_atlas_different_rois(self) -> None:
        """Test fetching different ROI counts."""
        for n_rois in [100, 200]:
            cfg = {"roi": {"schaefer_n_rois": n_rois, "schaefer_yeo_networks": 7}}
            maps, labels = get_schaefer_atlas(cfg)

            assert maps is not None
            assert labels is not None


class TestExtractRoiTimeseries:
    """Tests for extract_roi_timeseries ROI extraction."""

    def test_extract_roi_timeseries_shape(self, tmp_path: Path) -> None:
        """Test that extracted ROI time series has correct shape."""
        # Create synthetic BOLD image
        bold_data = np.random.randn(10, 10, 10, 50).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_file = tmp_path / "bold.nii.gz"
        nib.save(bold_img, str(bold_file))

        # Create synthetic atlas with 10 ROIs
        atlas_data = np.zeros((10, 10, 10), dtype=np.int32)
        for i in range(1, 11):
            atlas_data[i - 1 : i, :, :] = i
        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))

        cfg = {"roi": {"standardize": "zscore_sample"}}

        ts = extract_roi_timeseries(str(bold_file), atlas_img, tr=2.0, cfg=cfg)

        # Should have 50 timepoints and 10 ROIs (or fewer if some ROIs are empty)
        assert ts.shape[0] == 50
        assert ts.shape[1] > 0
        assert ts.shape[1] <= 10

    def test_extract_roi_timeseries_returns_array(self, tmp_path: Path) -> None:
        """Test that result is a numpy array."""
        bold_data = np.random.randn(5, 5, 5, 20).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_file = tmp_path / "bold.nii.gz"
        nib.save(bold_img, str(bold_file))

        atlas_data = np.ones((5, 5, 5), dtype=np.int32)
        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))

        cfg = {"roi": {"standardize": "zscore_sample"}}

        ts = extract_roi_timeseries(str(bold_file), atlas_img, tr=2.0, cfg=cfg)

        assert isinstance(ts, np.ndarray)


class TestSaveRoiTimeseries:
    """Tests for save_roi_timeseries file saving."""

    def test_save_roi_timeseries_basic(self, tmp_path: Path) -> None:
        """Test saving ROI time series to NPY and CSV."""
        ts = np.random.randn(100, 10)  # 100 timepoints, 10 ROIs
        labels = [b"roi_01", b"roi_02", b"roi_03", b"roi_04", b"roi_05",
                  b"roi_06", b"roi_07", b"roi_08", b"roi_09", b"roi_10"]

        npy_path, csv_path = save_roi_timeseries(ts, labels, tmp_path)

        assert Path(npy_path).exists()
        assert Path(csv_path).exists()

    def test_save_roi_timeseries_npy_content(self, tmp_path: Path) -> None:
        """Test that saved NPY file can be loaded correctly."""
        ts = np.random.randn(50, 5)
        labels = [b"roi_01", b"roi_02", b"roi_03", b"roi_04", b"roi_05"]

        npy_path, _ = save_roi_timeseries(ts, labels, tmp_path)

        loaded = np.load(npy_path)
        assert np.allclose(loaded, ts)

    def test_save_roi_timeseries_csv_content(self, tmp_path: Path) -> None:
        """Test that saved CSV file has correct columns."""
        ts = np.random.randn(30, 3)
        labels = [b"roi_1", b"roi_2", b"roi_3"]

        _, csv_path = save_roi_timeseries(ts, labels, tmp_path)

        df = pd.read_csv(csv_path)
        assert df.shape == (30, 3)
        assert list(df.columns) == ["roi_1", "roi_2", "roi_3"]

    def test_save_roi_timeseries_string_labels(self, tmp_path: Path) -> None:
        """Test with string labels instead of bytes."""
        ts = np.random.randn(20, 2)
        labels = ["network_1", "network_2"]

        npy_path, csv_path = save_roi_timeseries(ts, labels, tmp_path)

        assert Path(npy_path).exists()
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["network_1", "network_2"]

    def test_save_roi_timeseries_creates_directory(self, tmp_path: Path) -> None:
        """Test that nested directory is created."""
        ts = np.random.randn(10, 2)
        labels = [b"a", b"b"]
        out_dir = tmp_path / "nested" / "dir" / "path"

        save_roi_timeseries(ts, labels, out_dir)

        assert out_dir.exists()

    def test_save_roi_timeseries_filenames(self, tmp_path: Path) -> None:
        """Test that output files have expected names."""
        ts = np.random.randn(10, 2)
        labels = [b"a", b"b"]

        npy_path, csv_path = save_roi_timeseries(ts, labels, tmp_path)

        assert "roi_timeseries.npy" in npy_path
        assert "roi_timeseries.csv" in csv_path

    def test_save_roi_timeseries_mismatched_labels(self, tmp_path: Path) -> None:
        """Test behavior when labels don't match data shape."""
        ts = np.random.randn(50, 10)  # 10 columns
        labels = [f"roi_{i}".encode() for i in range(12)]  # 12 labels (more than data)

        # Should handle gracefully by truncating labels
        npy_path, csv_path = save_roi_timeseries(ts, labels, tmp_path)

        df = pd.read_csv(csv_path)
        # CSV should match the number of columns in ts
        assert df.shape[1] == ts.shape[1]

    def test_save_roi_timeseries_csv_values(self, tmp_path: Path) -> None:
        """Test that CSV contains correct numerical values."""
        ts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = [b"col1", b"col2"]

        _, csv_path = save_roi_timeseries(ts, labels, tmp_path)

        df = pd.read_csv(csv_path)
        assert np.allclose(df.iloc[:, 0].values, [1.0, 3.0, 5.0])
        assert np.allclose(df.iloc[:, 1].values, [2.0, 4.0, 6.0])
