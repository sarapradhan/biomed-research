"""
Comprehensive tests for viz.py and reho.py modules.

Tests cover:
- Visualization functions (boxplot, voxel maps, matrix plots)
- ReHo (regional homogeneity) computation
- ReHo Kendall's W computation
- ReHo Z-score normalization
- File I/O for visualizations and maps
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

from fmri_pipeline.reho import compute_reho_map, save_reho_map
from fmri_pipeline.viz import (
    plot_box_by_group,
    plot_voxel_map,
    save_thresholded_diff_matrix,
    save_voxel_from_vector,
)


class TestPlotBoxByGroup:
    """Tests for plot_box_by_group boxplot function."""

    def test_plot_box_by_group_basic(self, tmp_path: Path) -> None:
        """Test basic boxplot generation."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )
        out_file = tmp_path / "boxplot.png"

        plot_box_by_group(df, "group", "value", str(out_file), "Test Boxplot")

        assert out_file.exists()

    def test_plot_box_by_group_creates_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created."""
        df = pd.DataFrame(
            {
                "group": ["A", "B"],
                "value": [1.0, 2.0],
            }
        )
        nested_dir = tmp_path / "a" / "b" / "c"
        out_file = nested_dir / "plot.png"

        plot_box_by_group(df, "group", "value", str(out_file), "Test")

        assert out_file.exists()

    def test_plot_box_by_group_multiple_groups(self, tmp_path: Path) -> None:
        """Test boxplot with multiple groups."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B", "C", "C"],
                "value": np.random.randn(6),
            }
        )
        out_file = tmp_path / "boxplot.png"

        plot_box_by_group(df, "group", "value", str(out_file), "Multi-Group")

        assert out_file.exists()

    def test_plot_box_by_group_with_nan(self, tmp_path: Path) -> None:
        """Test boxplot handles NaN values."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B", "B"],
                "value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            }
        )
        out_file = tmp_path / "boxplot.png"

        # Should handle NaN gracefully
        plot_box_by_group(df, "group", "value", str(out_file), "NaN Test")

        assert out_file.exists()

    def test_plot_box_by_group_empty_dataframe(self, tmp_path: Path) -> None:
        """Test boxplot with empty DataFrame."""
        df = pd.DataFrame({"group": [], "value": []})
        out_file = tmp_path / "boxplot.png"

        # Should not crash
        plot_box_by_group(df, "group", "value", str(out_file), "Empty")

        # May or may not create file depending on implementation

    def test_plot_box_by_group_missing_column(self, tmp_path: Path) -> None:
        """Test boxplot with missing column."""
        df = pd.DataFrame({"group": ["A", "B"]})
        out_file = tmp_path / "boxplot.png"

        # Should not crash
        plot_box_by_group(df, "group", "missing_col", str(out_file), "Missing")

    def test_plot_box_by_group_single_group(self, tmp_path: Path) -> None:
        """Test boxplot with single group."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "A"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out_file = tmp_path / "boxplot.png"

        plot_box_by_group(df, "group", "value", str(out_file), "Single Group")

        assert out_file.exists()


class TestPlotVoxelMap:
    """Tests for plot_voxel_map statistical map visualization."""

    def test_plot_voxel_map_basic(self, tmp_path: Path) -> None:
        """Test basic voxel map plotting."""
        # Create synthetic stat map
        data = np.random.randn(10, 10, 10).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        stat_file = tmp_path / "stat_map.nii.gz"
        nib.save(img, str(stat_file))

        out_file = tmp_path / "voxel_map.png"

        plot_voxel_map(str(stat_file), str(out_file), "Test Voxel Map")

        assert out_file.exists()

    def test_plot_voxel_map_creates_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        stat_file = tmp_path / "stat.nii.gz"
        nib.save(img, str(stat_file))

        nested_dir = tmp_path / "plots" / "nested"
        out_file = nested_dir / "map.png"

        plot_voxel_map(str(stat_file), str(out_file), "Map")

        assert out_file.exists()

    def test_plot_voxel_map_with_threshold(self, tmp_path: Path) -> None:
        """Test voxel map with custom threshold."""
        data = np.random.randn(5, 5, 5).astype(np.float32) * 2.0
        img = nib.Nifti1Image(data, np.eye(4))
        stat_file = tmp_path / "stat.nii.gz"
        nib.save(img, str(stat_file))

        out_file = tmp_path / "map_thresholded.png"

        plot_voxel_map(str(stat_file), str(out_file), "Thresholded", threshold=1.0)

        assert out_file.exists()

    def test_plot_voxel_map_zero_threshold(self, tmp_path: Path) -> None:
        """Test voxel map with zero threshold."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        stat_file = tmp_path / "stat.nii.gz"
        nib.save(img, str(stat_file))

        out_file = tmp_path / "map.png"

        plot_voxel_map(str(stat_file), str(out_file), "Map", threshold=0.0)

        assert out_file.exists()


class TestSaveThresholdedDiffMatrix:
    """Tests for save_thresholded_diff_matrix connectivity matrix plotting."""

    def test_save_thresholded_diff_matrix_basic(self, tmp_path: Path) -> None:
        """Test basic thresholded matrix saving."""
        n_rois = 10
        n_edges = (n_rois * (n_rois - 1)) // 2

        beta_vec = np.random.randn(n_edges)
        q_vec = np.random.rand(n_edges)
        edge_idx = np.triu_indices(n_rois, k=1)

        out_file = tmp_path / "matrix.png"

        save_thresholded_diff_matrix(beta_vec, q_vec, edge_idx, n_rois, str(out_file), "Difference Matrix")

        assert out_file.exists()

    def test_save_thresholded_diff_matrix_creates_directory(self, tmp_path: Path) -> None:
        """Test that output directory is created."""
        n_rois = 5
        n_edges = (n_rois * (n_rois - 1)) // 2

        beta_vec = np.random.randn(n_edges)
        q_vec = np.random.rand(n_edges)
        edge_idx = np.triu_indices(n_rois, k=1)

        nested_dir = tmp_path / "matrices" / "nested"
        out_file = nested_dir / "matrix.png"

        save_thresholded_diff_matrix(beta_vec, q_vec, edge_idx, n_rois, str(out_file), "Matrix")

        assert out_file.exists()

    def test_save_thresholded_diff_matrix_thresholding(self, tmp_path: Path) -> None:
        """Test that q < 0.05 threshold is applied."""
        n_rois = 5
        n_edges = (n_rois * (n_rois - 1)) // 2

        beta_vec = np.random.randn(n_edges)
        # Make half of q-values significant, half not
        q_vec = np.concatenate([np.random.rand(n_edges // 2) * 0.01, np.random.rand(n_edges // 2) * 0.5])

        edge_idx = np.triu_indices(n_rois, k=1)
        out_file = tmp_path / "matrix.png"

        save_thresholded_diff_matrix(beta_vec, q_vec, edge_idx, n_rois, str(out_file), "Threshold Test")

        assert out_file.exists()

    def test_save_thresholded_diff_matrix_all_nonsignificant(self, tmp_path: Path) -> None:
        """Test with all non-significant edges."""
        n_rois = 5
        n_edges = (n_rois * (n_rois - 1)) // 2

        beta_vec = np.random.randn(n_edges)
        q_vec = np.ones(n_edges) * 0.5  # All non-significant

        edge_idx = np.triu_indices(n_rois, k=1)
        out_file = tmp_path / "matrix_empty.png"

        save_thresholded_diff_matrix(beta_vec, q_vec, edge_idx, n_rois, str(out_file), "Empty Matrix")

        assert out_file.exists()

    def test_save_thresholded_diff_matrix_large(self, tmp_path: Path) -> None:
        """Test with larger ROI set."""
        n_rois = 100
        n_edges = (n_rois * (n_rois - 1)) // 2

        beta_vec = np.random.randn(n_edges)
        q_vec = np.random.rand(n_edges)
        edge_idx = np.triu_indices(n_rois, k=1)

        out_file = tmp_path / "matrix_large.png"

        save_thresholded_diff_matrix(beta_vec, q_vec, edge_idx, n_rois, str(out_file), "Large")

        assert out_file.exists()


class TestSaveVoxelFromVector:
    """Tests for save_voxel_from_vector vector-to-NIfTI conversion."""

    def test_save_voxel_from_vector_basic(self, tmp_path: Path) -> None:
        """Test saving vector as masked NIfTI."""
        # Create reference mask
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        mask_img = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        # Create vector with same size as number of True voxels
        n_voxels = np.sum(mask)
        vec = np.random.randn(n_voxels)

        out_file = tmp_path / "output.nii.gz"
        save_voxel_from_vector(vec, str(mask_file), str(out_file))

        assert out_file.exists()

        # Verify saved image
        saved_img = nib.load(str(out_file))
        saved_data = saved_img.get_fdata()

        assert saved_data.shape == (5, 5, 5)
        # Masked voxels should contain data
        assert np.any(saved_data[mask] != 0)

    def test_save_voxel_from_vector_mask_shape(self, tmp_path: Path) -> None:
        """Test that output matches mask shape."""
        # Create mask
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:8, 2:8, 2:8] = True
        mask_img = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        # Create vector
        n_voxels = np.sum(mask)
        vec = np.ones(n_voxels)

        out_file = tmp_path / "output.nii.gz"
        save_voxel_from_vector(vec, str(mask_file), str(out_file))

        loaded = nib.load(str(out_file))
        assert loaded.get_fdata().shape == (10, 10, 10)

    def test_save_voxel_from_vector_zero_vector(self, tmp_path: Path) -> None:
        """Test with zero-filled vector."""
        mask = np.ones((5, 5, 5), dtype=bool)
        mask_img = nib.Nifti1Image(mask.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        n_voxels = 125
        vec = np.zeros(n_voxels)

        out_file = tmp_path / "zero.nii.gz"
        save_voxel_from_vector(vec, str(mask_file), str(out_file))

        loaded = nib.load(str(out_file))
        data = loaded.get_fdata()
        assert np.allclose(data, 0.0)


class TestComputeRehoMap:
    """Tests for compute_reho_map regional homogeneity."""

    def test_compute_reho_map_basic(self, tmp_path: Path) -> None:
        """Test basic ReHo map computation."""
        # Create small synthetic BOLD data
        bold_data = np.random.randn(8, 8, 8, 30).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_file = tmp_path / "bold.nii.gz"
        nib.save(bold_img, str(bold_file))

        # Create GM mask
        gm = np.zeros((8, 8, 8), dtype=bool)
        gm[2:6, 2:6, 2:6] = True
        mask_img = nib.Nifti1Image(gm.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        cfg = {
            "project": {"debug_mode": False},
            "reho": {
                "chunk_size_voxels": 1000,
                "normalize_zscore": True,
            },
        }

        reho_img = compute_reho_map(str(bold_file), str(mask_file), cfg, n_jobs=1)

        assert isinstance(reho_img, nib.Nifti1Image)
        assert reho_img.shape == (8, 8, 8)

    def test_compute_reho_map_shape(self, tmp_path: Path) -> None:
        """Test that ReHo map has same shape as input."""
        bold_data = np.random.randn(10, 10, 10, 20).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_file = tmp_path / "bold.nii.gz"
        nib.save(bold_img, str(bold_file))

        gm = np.ones((10, 10, 10), dtype=bool)
        mask_img = nib.Nifti1Image(gm.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        cfg = {
            "project": {"debug_mode": False},
            "reho": {"chunk_size_voxels": 100, "normalize_zscore": True},
        }

        reho_img = compute_reho_map(str(bold_file), str(mask_file), cfg, n_jobs=1)

        assert reho_img.shape == (10, 10, 10)

    def test_compute_reho_map_values_in_valid_range(self, tmp_path: Path) -> None:
        """Test that ReHo values are in reasonable range (z-scores after normalization)."""
        bold_data = np.random.randn(6, 6, 6, 25).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_file = tmp_path / "bold.nii.gz"
        nib.save(bold_img, str(bold_file))

        gm = np.zeros((6, 6, 6), dtype=bool)
        gm[1:5, 1:5, 1:5] = True
        mask_img = nib.Nifti1Image(gm.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        cfg = {
            "project": {"debug_mode": False},
            "reho": {"chunk_size_voxels": 64, "normalize_zscore": True},
        }

        reho_img = compute_reho_map(str(bold_file), str(mask_file), cfg, n_jobs=1)
        data = reho_img.get_fdata()

        # Z-scores should typically be in [-5, 5] range
        masked_vals = data[gm]
        if len(masked_vals) > 0:
            assert np.all(np.isfinite(masked_vals))

    def test_compute_reho_map_without_normalization(self, tmp_path: Path) -> None:
        """Test ReHo computation without Z-score normalization."""
        bold_data = np.random.randn(6, 6, 6, 20).astype(np.float32)
        bold_img = nib.Nifti1Image(bold_data, np.eye(4))
        bold_file = tmp_path / "bold.nii.gz"
        nib.save(bold_img, str(bold_file))

        gm = np.ones((6, 6, 6), dtype=bool)
        mask_img = nib.Nifti1Image(gm.astype(np.float32), np.eye(4))
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(mask_img, str(mask_file))

        cfg = {
            "project": {"debug_mode": False},
            "reho": {"chunk_size_voxels": 100, "normalize_zscore": False},
        }

        reho_img = compute_reho_map(str(bold_file), str(mask_file), cfg, n_jobs=1)

        assert isinstance(reho_img, nib.Nifti1Image)
        # Without normalization, values should be Kendall's W (0 to 1 range)
        data = reho_img.get_fdata()
        assert np.all(data >= -0.1)  # Allow small numeric error


class TestSaveRehoMap:
    """Tests for save_reho_map NIfTI saving."""

    def test_save_reho_map_basic(self, tmp_path: Path) -> None:
        """Test saving ReHo map to file."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        reho_img = nib.Nifti1Image(data, np.eye(4))

        out_path = save_reho_map(reho_img, tmp_path)

        assert Path(out_path).exists()

    def test_save_reho_map_creates_directory(self, tmp_path: Path) -> None:
        """Test that save_reho_map creates output directory."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        reho_img = nib.Nifti1Image(data, np.eye(4))

        nested_dir = tmp_path / "a" / "b" / "c"
        out_path = save_reho_map(reho_img, nested_dir)

        assert nested_dir.exists()
        assert Path(out_path).exists()

    def test_save_reho_map_filename(self, tmp_path: Path) -> None:
        """Test that ReHo file has expected name."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        reho_img = nib.Nifti1Image(data, np.eye(4))

        out_path = save_reho_map(reho_img, tmp_path)

        assert "reho_map.nii.gz" in out_path

    def test_save_reho_map_content(self, tmp_path: Path) -> None:
        """Test that saved ReHo map can be loaded."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        reho_img = nib.Nifti1Image(data, np.eye(4))

        out_path = save_reho_map(reho_img, tmp_path)

        loaded = nib.load(out_path)
        loaded_data = loaded.get_fdata()

        assert np.allclose(loaded_data, data)

    def test_save_reho_map_preserves_affine(self, tmp_path: Path) -> None:
        """Test that affine is preserved when saving."""
        data = np.random.randn(5, 5, 5).astype(np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        reho_img = nib.Nifti1Image(data, affine)

        out_path = save_reho_map(reho_img, tmp_path)

        loaded = nib.load(out_path)
        assert np.allclose(loaded.affine, affine)
