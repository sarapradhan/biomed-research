from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.isc import _corr_time, _loo_isc, _zscore_t


class TestZscoreT:
    """Tests for _zscore_t function."""

    def test_zscore_t_shape(self) -> None:
        """Test that z-scoring preserves shape."""
        x = np.random.randn(100, 50)
        result = _zscore_t(x)
        assert result.shape == x.shape

    def test_zscore_t_zero_mean(self) -> None:
        """Test that z-scored data has zero mean along time axis."""
        x = np.random.randn(100, 50)
        result = _zscore_t(x)
        means = np.mean(result, axis=0)
        assert np.allclose(means, 0.0, atol=1e-10)

    def test_zscore_t_unit_std(self) -> None:
        """Test that z-scored data has unit std along time axis."""
        x = np.random.randn(100, 50)
        result = _zscore_t(x)
        stds = np.std(result, axis=0)
        assert np.allclose(stds, 1.0, atol=1e-10)

    def test_zscore_t_constant_signal(self) -> None:
        """Test z-scoring of constant signal (edge case)."""
        x = np.ones((100, 50)) * 5.0  # Constant signal
        result = _zscore_t(x)
        # When std=0, should be replaced with 1.0, so result = (x - mean) / 1 = 0
        assert np.allclose(result, 0.0)

    def test_zscore_t_mixed_signals(self) -> None:
        """Test z-scoring with mixed constant and varying signals."""
        x = np.zeros((100, 50))
        x[:, :25] = np.random.randn(100, 25)  # Varying
        x[:, 25:] = 3.0  # Constant

        result = _zscore_t(x)

        # Varying signals should have unit std
        assert np.allclose(np.std(result[:, :25], axis=0), 1.0, atol=1e-10)
        # Constant signals should be zero
        assert np.allclose(result[:, 25:], 0.0)


class TestCorrTime:
    """Tests for _corr_time function."""

    def test_corr_time_identical(self) -> None:
        """Test correlation of identical arrays."""
        a = np.random.randn(100, 50)
        b = a.copy()
        result = _corr_time(a, b)

        # Should be 1.0 for all voxels
        assert np.allclose(result, 1.0, atol=1e-10)

    def test_corr_time_random(self) -> None:
        """Test correlation of independent random arrays."""
        a = np.random.randn(100, 50)
        b = np.random.randn(100, 50)
        result = _corr_time(a, b)

        # Should be close to 0 on average
        assert np.isclose(np.mean(result), 0.0, atol=0.1)
        # Should be in [-1, 1] range
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_corr_time_negation(self) -> None:
        """Test correlation with negated signal."""
        a = np.random.randn(100, 50)
        b = -a  # Perfect negative correlation
        result = _corr_time(a, b)

        # Should be -1.0 for all voxels
        assert np.allclose(result, -1.0, atol=1e-10)

    def test_corr_time_shape(self) -> None:
        """Test that output shape matches number of voxels."""
        a = np.random.randn(100, 50)
        b = np.random.randn(100, 50)
        result = _corr_time(a, b)

        assert result.shape == (50,)

    def test_corr_time_linear_relationship(self) -> None:
        """Test correlation of linearly related signals."""
        a = np.random.randn(100, 50)
        b = 2.0 * a + 1.0  # Linear transformation

        result = _corr_time(a, b)

        # Should be 1.0 (linear relationship with positive slope)
        assert np.allclose(result, 1.0, atol=1e-10)


class TestLooISC:
    """Tests for _loo_isc function (leave-one-out ISC)."""

    def test_loo_isc_shape(self) -> None:
        """Test output shapes for leave-one-out ISC."""
        ts_list = [
            np.random.randn(100, 50),
            np.random.randn(100, 50),
            np.random.randn(100, 50),
            np.random.randn(100, 50),
        ]

        loo_arr, mean_map = _loo_isc(ts_list)

        # loo_arr: [n_subjects, n_voxels]
        assert loo_arr.shape == (4, 50)
        # mean_map: [n_voxels]
        assert mean_map.shape == (50,)

    def test_loo_isc_mean_correct(self) -> None:
        """Test that mean_map is correct mean of loo_arr."""
        ts_list = [
            np.random.randn(100, 50),
            np.random.randn(100, 50),
            np.random.randn(100, 50),
        ]

        loo_arr, mean_map = _loo_isc(ts_list)

        # mean_map should equal mean of loo_arr across subjects
        expected_mean = np.mean(loo_arr, axis=0)
        assert np.allclose(mean_map, expected_mean)

    def test_loo_isc_leaves_one_out(self) -> None:
        """Test that LOO correctly excludes each subject."""
        # Create simple case: identical subjects
        # Use a ramp so std > 0; identical across subjects → LOO corr = 1.0
        ts = np.outer(np.linspace(0, 1, 100), np.ones(10))
        ts_list = [ts, ts, ts, ts]

        loo_arr, _ = _loo_isc(ts_list)

        # All subjects identical: correlation of each with mean of others = 1.0
        assert np.allclose(loo_arr, 1.0, atol=1e-10)

    def test_loo_isc_two_subjects(self) -> None:
        """Test LOO ISC with minimum viable number of subjects."""
        ts_list = [
            np.random.randn(100, 50),
            np.random.randn(100, 50),
        ]

        loo_arr, mean_map = _loo_isc(ts_list)

        assert loo_arr.shape == (2, 50)
        assert mean_map.shape == (50,)
        # Each subject correlated with the other
        assert np.all(loo_arr >= -1.0) and np.all(loo_arr <= 1.0)

    def test_loo_isc_range(self) -> None:
        """Test that ISC values are in valid range."""
        ts_list = [
            np.random.randn(100, 50),
            np.random.randn(100, 50),
            np.random.randn(100, 50),
            np.random.randn(100, 50),
        ]

        loo_arr, mean_map = _loo_isc(ts_list)

        # ISC should be in [-1, 1]
        assert np.all(loo_arr >= -1.0) and np.all(loo_arr <= 1.0)
        assert np.all(mean_map >= -1.0) and np.all(mean_map <= 1.0)

    def test_loo_isc_many_subjects(self) -> None:
        """Test LOO ISC with many subjects."""
        n_subjects = 10
        ts_list = [np.random.randn(100, 50) for _ in range(n_subjects)]

        loo_arr, mean_map = _loo_isc(ts_list)

        assert loo_arr.shape == (n_subjects, 50)
        assert mean_map.shape == (50,)

    def test_loo_isc_deterministic(self) -> None:
        """Test that same input produces same output."""
        # Create deterministic data
        np.random.seed(42)
        ts_list = [np.random.randn(100, 50) for _ in range(4)]

        loo_arr1, mean_map1 = _loo_isc([t.copy() for t in ts_list])
        loo_arr2, mean_map2 = _loo_isc([t.copy() for t in ts_list])

        assert np.allclose(loo_arr1, loo_arr2)
        assert np.allclose(mean_map1, mean_map2)

    def test_loo_isc_positive_correlation_increases_mean(self) -> None:
        """Test that more similar subjects increase ISC values."""
        # Create highly correlated subjects
        base = np.random.randn(100, 50)
        ts_list_corr = [
            base,
            base + 0.1 * np.random.randn(100, 50),
            base + 0.1 * np.random.randn(100, 50),
        ]

        # Create uncorrelated subjects
        ts_list_uncorr = [
            np.random.randn(100, 50),
            np.random.randn(100, 50),
            np.random.randn(100, 50),
        ]

        _, mean_corr = _loo_isc(ts_list_corr)
        _, mean_uncorr = _loo_isc(ts_list_uncorr)

        # Correlated subjects should have higher ISC
        assert np.mean(mean_corr) > np.mean(mean_uncorr)


class TestISCIntegration:
    """Integration tests for ISC functions."""

    def test_zscore_and_corr_chain(self) -> None:
        """Test chaining zscore and correlation."""
        a = np.random.randn(100, 50)
        b = np.random.randn(100, 50)

        az = _zscore_t(a)
        bz = _zscore_t(b)
        # Manual correlation
        manual_corr = np.mean(az * bz, axis=0)

        # Using _corr_time
        func_corr = _corr_time(a, b)

        assert np.allclose(manual_corr, func_corr)

    def test_loo_uses_zscore_and_corr(self) -> None:
        """Test that LOO ISC uses z-scoring and correlation internally."""
        # Use simple case where we can verify manually
        ts = np.random.randn(100, 10)
        ts_list = [ts, ts]  # Identical - should give ISC ≈ 1

        loo_arr, _ = _loo_isc(ts_list)

        # Each should correlate with itself
        assert np.allclose(loo_arr, 1.0, atol=1e-10)
