from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.connectivity import (
    dynamic_fc_summary,
    sliding_windows,
    static_fc,
)


class TestStaticFC:
    """Tests for static_fc function."""

    def test_static_fc_shape(self) -> None:
        """Test that static_fc returns correct shape."""
        roi_ts = np.random.randn(100, 10)
        result = static_fc(roi_ts)
        assert result.shape == (10, 10)

    def test_static_fc_diagonal_zeros(self) -> None:
        """Test that diagonal is zero."""
        roi_ts = np.random.randn(100, 10)
        result = static_fc(roi_ts)
        assert np.allclose(np.diag(result), 0.0)

    def test_static_fc_symmetry(self) -> None:
        """Test that FC matrix is symmetric."""
        roi_ts = np.random.randn(100, 10)
        result = static_fc(roi_ts)
        assert np.allclose(result, result.T)

    def test_static_fc_range(self) -> None:
        """Test that Fisher-z values are in reasonable range."""
        roi_ts = np.random.randn(100, 10)
        result = static_fc(roi_ts)
        # Fisher-z transform of [-0.999999, 0.999999] is roughly [-8, 8]
        # Allow some margin for numerical precision
        assert np.all(np.abs(result) <= 10.0)

    def test_static_fc_identical_signal(self) -> None:
        """Test FC of identical signals."""
        # Create two identical signals and one different
        x = np.random.randn(100, 1)
        roi_ts = np.hstack([x, x, np.random.randn(100, 2)])
        result = static_fc(roi_ts)
        # FC between columns 0 and 1 should be high (they're identical)
        # But diagonal should be zero
        assert result[0, 1] > 0.9
        assert result[1, 0] > 0.9
        assert np.abs(result[0, 0]) < 1e-10


class TestSlidingWindows:
    """Tests for sliding_windows function."""

    def test_sliding_windows_count(self) -> None:
        """Test correct number of windows for given input."""
        roi_ts = np.random.randn(100, 10)
        window_trs = 20
        step_trs = 5
        windows = sliding_windows(roi_ts, window_trs, step_trs)
        # Expected count: floor((100 - 20) / 5) + 1 = floor(16) + 1 = 17
        expected_count = (100 - window_trs) // step_trs + 1
        assert len(windows) == expected_count

    def test_sliding_windows_count_tight(self) -> None:
        """Test window count with exact boundary."""
        roi_ts = np.random.randn(40, 10)
        window_trs = 20
        step_trs = 10
        windows = sliding_windows(roi_ts, window_trs, step_trs)
        # Expected: windows at [0-20), [10-30), [20-40)
        assert len(windows) == 3

    def test_sliding_windows_shape(self) -> None:
        """Test that each window has correct FC matrix shape."""
        roi_ts = np.random.randn(100, 10)
        windows = sliding_windows(roi_ts, 20, 5)
        for window_mat in windows:
            assert window_mat.shape == (10, 10)

    def test_sliding_windows_empty(self) -> None:
        """Test behavior with window larger than data."""
        roi_ts = np.random.randn(10, 10)
        windows = sliding_windows(roi_ts, 20, 5)
        assert len(windows) == 0

    def test_sliding_windows_single_window(self) -> None:
        """Test with exactly one window."""
        roi_ts = np.random.randn(20, 10)
        windows = sliding_windows(roi_ts, 20, 5)
        assert len(windows) == 1


class TestDynamicFCSummary:
    """Tests for dynamic_fc_summary function."""

    def test_dynamic_fc_summary_shapes(self) -> None:
        """Test that mean and std matrices have correct shapes."""
        roi_ts = np.random.randn(100, 10)
        cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}
        mean, std, windows = dynamic_fc_summary(roi_ts, cfg)
        assert mean.shape == (10, 10)
        assert std.shape == (10, 10)
        assert len(windows) > 0

    def test_dynamic_fc_summary_empty_windows(self) -> None:
        """Test behavior when no windows can be created."""
        roi_ts = np.random.randn(10, 5)
        cfg = {"dynamic_fc": {"window_trs": 50, "step_trs": 5}}
        mean, std, windows = dynamic_fc_summary(roi_ts, cfg)
        assert mean.shape == (5, 5)
        assert std.shape == (5, 5)
        assert len(windows) == 0
        # Should return zeros
        assert np.allclose(mean, 0.0)
        assert np.allclose(std, 0.0)

    def test_dynamic_fc_summary_diagonal_zeros(self) -> None:
        """Test that diagonal is zero in mean and std."""
        roi_ts = np.random.randn(100, 10)
        cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}
        mean, std, _ = dynamic_fc_summary(roi_ts, cfg)
        assert np.allclose(np.diag(mean), 0.0)
        assert np.allclose(np.diag(std), 0.0)

    def test_dynamic_fc_summary_symmetry(self) -> None:
        """Test that mean and std matrices are symmetric."""
        roi_ts = np.random.randn(100, 10)
        cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}
        mean, std, _ = dynamic_fc_summary(roi_ts, cfg)
        assert np.allclose(mean, mean.T)
        assert np.allclose(std, std.T)

    def test_dynamic_fc_summary_std_non_negative(self) -> None:
        """Test that standard deviation is non-negative."""
        roi_ts = np.random.randn(100, 10)
        cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}
        _, std, _ = dynamic_fc_summary(roi_ts, cfg)
        # std should be non-negative (where not masked)
        off_diag = std[np.triu_indices_from(std, k=1)]
        assert np.all(off_diag >= 0)

    def test_dynamic_fc_summary_single_window(self) -> None:
        """Test with data allowing exactly one window."""
        roi_ts = np.random.randn(30, 10)
        cfg = {"dynamic_fc": {"window_trs": 30, "step_trs": 5}}
        mean, std, windows = dynamic_fc_summary(roi_ts, cfg)
        assert len(windows) == 1
        # Mean should match the single window's FC
        assert np.allclose(mean, windows[0])
        # Std of one window is zero
        assert np.allclose(std, 0.0)

    def test_dynamic_fc_summary_multiple_windows(self) -> None:
        """Test with multiple windows shows variance."""
        # Create data with distinct patterns in different windows
        roi_ts = np.random.randn(100, 10)
        cfg = {"dynamic_fc": {"window_trs": 20, "step_trs": 5}}
        mean, std, windows = dynamic_fc_summary(roi_ts, cfg)
        assert len(windows) > 1
        # If windows have different FC values, std should be > 0
        # (not guaranteed but very likely with random data)
        off_diag_std = std[np.triu_indices_from(std, k=1)]
        assert np.any(off_diag_std > 1e-6)
