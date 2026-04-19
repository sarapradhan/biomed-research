"""
Comprehensive tests for preprocessing.py module.

Tests cover:
- Friston-24 confound expansion (6 → 24 columns)
- WM/CSF confound selection
- Scrubbing mask generation
- Motion exclusion thresholds
- DVARS computation
- tSNR computation
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

from fmri_pipeline.preprocessing import (
    build_friston24,
    build_scrub_mask,
    compute_dvars,
    compute_motion_exclusion,
    compute_tsnr,
    select_wm_csf,
)


class TestBuildFriston24:
    """Tests for build_friston24 confound expansion."""

    def test_friston24_shape(self) -> None:
        """Test that Friston-24 output has exactly 24 columns."""
        n = 100
        df = pd.DataFrame(
            {
                "trans_x": np.random.randn(n),
                "trans_y": np.random.randn(n),
                "trans_z": np.random.randn(n),
                "rot_x": np.random.randn(n),
                "rot_y": np.random.randn(n),
                "rot_z": np.random.randn(n),
            }
        )
        result = build_friston24(df)
        assert result.shape == (n, 24)

    def test_friston24_columns(self) -> None:
        """Test that all expected column names are present."""
        n = 50
        df = pd.DataFrame(
            {
                "trans_x": np.random.randn(n),
                "trans_y": np.random.randn(n),
                "trans_z": np.random.randn(n),
                "rot_x": np.random.randn(n),
                "rot_y": np.random.randn(n),
                "rot_z": np.random.randn(n),
            }
        )
        result = build_friston24(df)

        # Check column names
        expected_parts = ["trans", "rot", "d_trans", "d_rot", "sq_trans", "sq_rot", "sq_d_trans", "sq_d_rot"]
        for part in expected_parts:
            matching = [c for c in result.columns if part in c]
            assert len(matching) > 0, f"No column containing '{part}'"

    def test_friston24_preserves_motion(self) -> None:
        """Test that original motion parameters are preserved in output."""
        n = 50
        motion_vals = {
            "trans_x": np.random.randn(n),
            "trans_y": np.random.randn(n),
            "trans_z": np.random.randn(n),
            "rot_x": np.random.randn(n),
            "rot_y": np.random.randn(n),
            "rot_z": np.random.randn(n),
        }
        df = pd.DataFrame(motion_vals)
        result = build_friston24(df)

        # First 6 columns should be original motion
        for i, col in enumerate(["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]):
            assert np.allclose(result.iloc[:, i].values, motion_vals[col])

    def test_friston24_with_nans(self) -> None:
        """Test that NaNs are filled with zeros."""
        n = 50
        df = pd.DataFrame(
            {
                "trans_x": np.concatenate([np.random.randn(40), [np.nan] * 10]),
                "trans_y": np.random.randn(n),
                "trans_z": np.random.randn(n),
                "rot_x": np.random.randn(n),
                "rot_y": np.random.randn(n),
                "rot_z": np.random.randn(n),
            }
        )
        result = build_friston24(df)

        # Should not contain NaNs
        assert not np.any(np.isnan(result.values))

    def test_friston24_missing_column_raises(self) -> None:
        """Test that missing motion columns raise ValueError."""
        df = pd.DataFrame(
            {
                "trans_x": np.random.randn(50),
                "trans_y": np.random.randn(50),
                # Missing trans_z and rotation columns
            }
        )
        with pytest.raises(ValueError, match="Missing motion columns"):
            build_friston24(df)

    def test_friston24_derivatives(self) -> None:
        """Test that temporal derivatives are computed correctly."""
        # Create simple linearly increasing motion
        n = 50
        motion = np.arange(n, dtype=float)
        df = pd.DataFrame(
            {
                "trans_x": motion,
                "trans_y": motion,
                "trans_z": motion,
                "rot_x": motion,
                "rot_y": motion,
                "rot_z": motion,
            }
        )
        result = build_friston24(df)

        # First derivative of linear signal should be constant
        # Column 6 should be d_trans_x
        deriv_col = result["d_trans_x"].values
        # First element is NaN (filled to 0), rest should be ~1.0
        assert np.isclose(deriv_col[1], 1.0, atol=1e-5)


class TestSelectWmCsf:
    """Tests for select_wm_csf confound selection."""

    def test_select_wm_csf_basic(self) -> None:
        """Test selecting WM and CSF confounds."""
        df = pd.DataFrame(
            {
                "white_matter": np.random.randn(50),
                "csf": np.random.randn(50),
                "trans_x": np.random.randn(50),
            }
        )
        cfg = {
            "preprocessing": {
                "confounds": {
                    "include_wm_csf": True,
                    "wm_columns": ["white_matter", "wm"],
                    "csf_columns": ["csf"],
                }
            }
        }
        result = select_wm_csf(df, cfg)
        assert result.shape[1] == 2
        assert "white_matter" in result.columns
        assert "csf" in result.columns

    def test_select_wm_csf_disabled(self) -> None:
        """Test that disabled WM/CSF selection returns empty DataFrame."""
        df = pd.DataFrame(
            {
                "white_matter": np.random.randn(50),
                "csf": np.random.randn(50),
            }
        )
        cfg = {
            "preprocessing": {
                "confounds": {
                    "include_wm_csf": False,
                }
            }
        }
        result = select_wm_csf(df, cfg)
        assert result.shape[1] == 0

    def test_select_wm_csf_with_nans(self) -> None:
        """Test that NaNs in confounds are filled with zeros."""
        df = pd.DataFrame(
            {
                "white_matter": np.concatenate([np.random.randn(40), [np.nan] * 10]),
                "csf": np.random.randn(50),
            }
        )
        cfg = {
            "preprocessing": {
                "confounds": {
                    "include_wm_csf": True,
                    "wm_columns": ["white_matter"],
                    "csf_columns": ["csf"],
                }
            }
        }
        result = select_wm_csf(df, cfg)
        assert not np.any(np.isnan(result.values))

    def test_select_wm_csf_missing_column(self) -> None:
        """Test behavior when requested column doesn't exist."""
        df = pd.DataFrame({"some_col": np.random.randn(50)})
        cfg = {
            "preprocessing": {
                "confounds": {
                    "include_wm_csf": True,
                    "wm_columns": ["white_matter"],
                    "csf_columns": ["csf"],
                }
            }
        }
        result = select_wm_csf(df, cfg)
        # Should return empty DataFrame if no columns match
        assert result.shape[1] == 0


class TestBuildScrubMask:
    """Tests for build_scrub_mask framewise displacement censoring."""

    def test_scrub_mask_basic(self) -> None:
        """Test basic scrubbing with FD threshold."""
        df = pd.DataFrame(
            {
                "framewise_displacement": np.array([0.1, 0.2, 0.8, 0.3, 0.15, 0.05]),
            }
        )
        keep, fd = build_scrub_mask(df, fd_threshold=0.5)

        assert keep.shape == (6,)
        assert np.array_equal(keep, np.array([True, True, False, True, True, True]))
        assert fd.shape == (6,)

    def test_scrub_mask_all_kept(self) -> None:
        """Test when all frames pass FD threshold."""
        df = pd.DataFrame(
            {
                "framewise_displacement": np.array([0.1, 0.2, 0.15, 0.05, 0.1]),
            }
        )
        keep, fd = build_scrub_mask(df, fd_threshold=1.0)

        assert np.all(keep)

    def test_scrub_mask_all_censored(self) -> None:
        """Test when all frames are censored."""
        df = pd.DataFrame(
            {
                "framewise_displacement": np.array([0.8, 0.9, 1.5, 2.0, 1.2]),
            }
        )
        keep, fd = build_scrub_mask(df, fd_threshold=0.5)

        assert not np.any(keep)

    def test_scrub_mask_missing_fd_column(self) -> None:
        """Test behavior when FD column is missing."""
        df = pd.DataFrame({"other_col": np.random.randn(50)})
        keep, fd = build_scrub_mask(df, fd_threshold=0.5)

        # Should create zero-filled FD series
        assert np.all(fd == 0.0)
        assert np.all(keep)  # All frames kept with zero FD

    def test_scrub_mask_with_nans(self) -> None:
        """Test FD NaNs are treated as zero (kept)."""
        df = pd.DataFrame(
            {
                "framewise_displacement": np.array([0.1, np.nan, 0.8, np.nan, 0.15]),
            }
        )
        keep, fd = build_scrub_mask(df, fd_threshold=0.5)

        # NaNs filled to 0, so only index 2 is censored
        assert np.array_equal(keep, np.array([True, True, False, True, True]))


class TestComputeMotionExclusion:
    """Tests for compute_motion_exclusion threshold evaluation."""

    def test_motion_exclusion_basic(self) -> None:
        """Test motion exclusion computation with normal motion."""
        df = pd.DataFrame(
            {
                "trans_x": np.random.randn(100) * 0.1,
                "trans_y": np.random.randn(100) * 0.1,
                "trans_z": np.random.randn(100) * 0.1,
                "rot_x": np.random.randn(100) * 0.01,
                "rot_y": np.random.randn(100) * 0.01,
                "rot_z": np.random.randn(100) * 0.01,
                "framewise_displacement": np.random.rand(100) * 0.2,
            }
        )
        keep = np.ones(100, dtype=bool)
        cfg = {
            "preprocessing": {
                "scrubbing": {
                    "exclude_percent_censored": 0.3,
                    "exclude_max_motion_mm": 3.0,
                    "exclude_max_rotation_deg": 3.0,
                }
            }
        }
        result = compute_motion_exclusion(df, keep, cfg)

        assert isinstance(result, dict)
        assert "percent_scrubbed" in result
        assert "max_translation_mm" in result
        assert "max_rotation_deg" in result
        assert "exclude" in result

    def test_motion_exclusion_high_censoring(self) -> None:
        """Test exclusion when censoring exceeds threshold."""
        df = pd.DataFrame(
            {
                "trans_x": np.zeros(100),
                "trans_y": np.zeros(100),
                "trans_z": np.zeros(100),
                "rot_x": np.zeros(100),
                "rot_y": np.zeros(100),
                "rot_z": np.zeros(100),
            }
        )
        # Censor 50% of frames
        keep = np.concatenate([np.ones(50), np.zeros(50)]).astype(bool)

        cfg = {
            "preprocessing": {
                "scrubbing": {
                    "exclude_percent_censored": 0.3,  # Threshold is 30%
                    "exclude_max_motion_mm": 10.0,
                    "exclude_max_rotation_deg": 10.0,
                }
            }
        }
        result = compute_motion_exclusion(df, keep, cfg)

        assert result["exclude"] is True

    def test_motion_exclusion_high_translation(self) -> None:
        """Test exclusion when translation exceeds threshold."""
        df = pd.DataFrame(
            {
                "trans_x": np.concatenate([np.zeros(50), np.ones(50) * 5.0]),  # 5mm max
                "trans_y": np.zeros(100),
                "trans_z": np.zeros(100),
                "rot_x": np.zeros(100),
                "rot_y": np.zeros(100),
                "rot_z": np.zeros(100),
            }
        )
        keep = np.ones(100, dtype=bool)

        cfg = {
            "preprocessing": {
                "scrubbing": {
                    "exclude_percent_censored": 0.5,
                    "exclude_max_motion_mm": 3.0,  # Threshold is 3mm
                    "exclude_max_rotation_deg": 10.0,
                }
            }
        }
        result = compute_motion_exclusion(df, keep, cfg)

        assert result["max_translation_mm"] > 3.0
        assert result["exclude"] is True

    def test_motion_exclusion_radians_to_degrees(self) -> None:
        """Test rotation is correctly converted from radians to degrees."""
        # 0.1 radians = ~5.73 degrees
        df = pd.DataFrame(
            {
                "trans_x": np.zeros(100),
                "trans_y": np.zeros(100),
                "trans_z": np.zeros(100),
                "rot_x": np.ones(100) * 0.1,
                "rot_y": np.zeros(100),
                "rot_z": np.zeros(100),
            }
        )
        keep = np.ones(100, dtype=bool)

        cfg = {
            "preprocessing": {
                "scrubbing": {
                    "exclude_percent_censored": 0.5,
                    "exclude_max_motion_mm": 10.0,
                    "exclude_max_rotation_deg": 3.0,  # Threshold 3 degrees
                }
            }
        }
        result = compute_motion_exclusion(df, keep, cfg)

        # 0.1 rad * 180/pi ~= 5.73 degrees, should exceed 3.0
        assert result["max_rotation_deg"] > 3.0
        assert result["exclude"] is True


class TestComputeDvars:
    """Tests for DVARS (temporal variance) computation."""

    def test_dvars_constant_volume(self) -> None:
        """Test DVARS of constant 4D array is zero."""
        arr = np.ones((10, 10, 10, 50)) * 100.0
        dvars = compute_dvars(arr)
        assert np.isclose(dvars, 0.0, atol=1e-5)

    def test_dvars_random_data(self) -> None:
        """Test DVARS returns positive value for random data."""
        arr = np.random.randn(10, 10, 10, 50)
        dvars = compute_dvars(arr)
        assert dvars > 0.0

    def test_dvars_shape(self) -> None:
        """Test DVARS works with various 4D shapes."""
        for shape in [(5, 5, 5, 20), (10, 10, 10, 100), (20, 20, 20, 50)]:
            arr = np.random.randn(*shape)
            dvars = compute_dvars(arr)
            assert isinstance(dvars, float)
            assert dvars >= 0.0

    def test_dvars_increasing_trend(self) -> None:
        """Test DVARS increases with temporal variation."""
        # Constant data
        arr_const = np.ones((10, 10, 10, 50)) * 100.0
        dvars_const = compute_dvars(arr_const)

        # Linearly increasing data (high temporal variation)
        arr_linear = np.arange(50).reshape(1, 1, 1, 50) * np.ones((10, 10, 10, 1))
        dvars_linear = compute_dvars(arr_linear)

        assert dvars_linear > dvars_const

    def test_dvars_empty_array(self) -> None:
        """Test DVARS returns 0 for empty array."""
        arr = np.array([]).reshape(0, 0, 0, 0)
        dvars = compute_dvars(arr)
        assert dvars == 0.0


class TestComputeTsnr:
    """Tests for temporal signal-to-noise ratio computation."""

    def test_tsnr_constant_signal(self) -> None:
        """Test tSNR of constant signal with noise is high."""
        arr = np.ones((10, 10, 10, 50)) * 100.0
        arr += np.random.randn(10, 10, 10, 50) * 0.1  # Small noise
        mask = np.ones((10, 10, 10), dtype=bool)

        tsnr = compute_tsnr(arr, mask)
        assert tsnr > 10.0  # Should be quite high

    def test_tsnr_high_noise(self) -> None:
        """Test tSNR decreases with higher noise."""
        # Low noise version
        arr_low_noise = np.ones((10, 10, 10, 50)) * 100.0
        arr_low_noise += np.random.randn(10, 10, 10, 50) * 1.0
        mask = np.ones((10, 10, 10), dtype=bool)
        tsnr_low = compute_tsnr(arr_low_noise, mask)

        # High noise version
        arr_high_noise = np.ones((10, 10, 10, 50)) * 100.0
        arr_high_noise += np.random.randn(10, 10, 10, 50) * 10.0
        tsnr_high = compute_tsnr(arr_high_noise, mask)

        # tSNR should be lower with higher noise
        assert tsnr_low > tsnr_high

    def test_tsnr_with_mask(self) -> None:
        """Test tSNR only includes voxels in mask."""
        # Two halves with very different SNR so tSNR differs between masks
        arr = np.zeros((10, 10, 10, 50))
        arr[:5, :, :, :] = 100.0 + np.random.randn(5, 10, 10, 50) * 1.0   # SNR ~100
        arr[5:, :, :, :] = 10.0 + np.random.randn(5, 10, 10, 50) * 10.0   # SNR ~1

        # Full mask (mean of ~100 and ~1 SNR voxels)
        mask_full = np.ones((10, 10, 10), dtype=bool)
        tsnr_full = compute_tsnr(arr, mask_full)

        # Half mask (only high-SNR voxels)
        mask_half = np.zeros((10, 10, 10), dtype=bool)
        mask_half[:5, :, :] = True
        tsnr_half = compute_tsnr(arr, mask_half)

        # tSNR should differ substantially between the two masks
        assert not np.isclose(tsnr_full, tsnr_half, rtol=0.1)

    def test_tsnr_empty_mask(self) -> None:
        """Test tSNR returns 0 for empty mask."""
        arr = np.random.randn(10, 10, 10, 50)
        mask = np.zeros((10, 10, 10), dtype=bool)

        tsnr = compute_tsnr(arr, mask)
        assert tsnr == 0.0

    def test_tsnr_single_timepoint(self) -> None:
        """Test tSNR with single timepoint (std=0) returns NaN mean handled."""
        arr = np.ones((10, 10, 10, 1)) * 100.0
        mask = np.ones((10, 10, 10), dtype=bool)

        tsnr = compute_tsnr(arr, mask)
        # With single timepoint, std=0, output should be 0 or NaN handled gracefully
        assert np.isfinite(tsnr)
