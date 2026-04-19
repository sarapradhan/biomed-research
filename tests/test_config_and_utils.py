"""
Comprehensive tests for config.py and utils.py modules.

Tests cover:
- YAML config loading and directory creation
- Logging setup
- Random seed management
- JSON serialization round-trips
- Run key/basename generation
- Metric path construction
- Upper triangle vector extraction
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.config import load_config
from fmri_pipeline.utils import (
    metric_path,
    run_basename,
    save_json,
    set_global_seed,
    setup_logging,
    upper_triangle_vector,
)


class TestLoadConfig:
    """Tests for load_config YAML loading."""

    def test_load_config_basic(self, tmp_path: Path) -> None:
        """Test loading a basic YAML config file."""
        config_content = """
paths:
  output_root: /tmp/test_output
  cache_dir: /tmp/test_cache
  logs_dir: /tmp/test_logs
project:
  name: test_project
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        cfg = load_config(str(config_file))

        assert isinstance(cfg, dict)
        assert "paths" in cfg
        assert cfg["paths"]["output_root"] == "/tmp/test_output"

    def test_load_config_creates_directories(self, tmp_path: Path) -> None:
        """Test that load_config creates required directories."""
        out_dir = tmp_path / "output"
        cache_dir = tmp_path / "cache"
        logs_dir = tmp_path / "logs"

        config_content = f"""
paths:
  output_root: {str(out_dir)}
  cache_dir: {str(cache_dir)}
  logs_dir: {str(logs_dir)}
project:
  name: test
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        cfg = load_config(str(config_file))

        assert out_dir.exists()
        assert cache_dir.exists()
        assert logs_dir.exists()

    def test_load_config_nested_paths(self, tmp_path: Path) -> None:
        """Test that nested path creation works."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"

        config_content = f"""
paths:
  output_root: {str(nested_dir)}
  cache_dir: {str(nested_dir / "cache")}
  logs_dir: {str(nested_dir / "logs")}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        cfg = load_config(str(config_file))

        assert nested_dir.exists()
        assert (nested_dir / "cache").exists()
        assert (nested_dir / "logs").exists()

    def test_load_config_idempotent(self, tmp_path: Path) -> None:
        """Test that loading config twice doesn't cause issues."""
        out_dir = tmp_path / "output"

        config_content = f"""
paths:
  output_root: {str(out_dir)}
  cache_dir: {str(out_dir / "cache")}
  logs_dir: {str(out_dir / "logs")}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        cfg1 = load_config(str(config_file))
        cfg2 = load_config(str(config_file))

        assert cfg1 == cfg2

    def test_load_config_preserves_content(self, tmp_path: Path) -> None:
        """Test that all YAML content is preserved."""
        config_content = """
paths:
  output_root: /tmp/out
  cache_dir: /tmp/cache
  logs_dir: /tmp/logs
preprocessing:
  smoothing_fwhm_mm: 6.0
  confounds:
    include_wm_csf: true
stats:
  diagnosis_column: diagnosis
  covariates:
    - age
    - sex
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        cfg = load_config(str(config_file))

        assert cfg["preprocessing"]["smoothing_fwhm_mm"] == 6.0
        assert cfg["preprocessing"]["confounds"]["include_wm_csf"] is True
        assert "diagnosis_column" in cfg["stats"]
        assert isinstance(cfg["stats"]["covariates"], list)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_creates_logger(self, tmp_path: Path) -> None:
        """Test that setup_logging returns a logger."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(str(log_dir), "test")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"

    def test_setup_logging_creates_directory(self, tmp_path: Path) -> None:
        """Test that log directory is created."""
        log_dir = tmp_path / "nested" / "logs"
        assert not log_dir.exists()

        logger = setup_logging(str(log_dir), "test")

        assert log_dir.exists()

    def test_setup_logging_creates_file(self, tmp_path: Path) -> None:
        """Test that log file is created."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(str(log_dir), "test")

        log_file = log_dir / "test.log"
        assert log_file.exists()

    def test_setup_logging_writes_messages(self, tmp_path: Path) -> None:
        """Test that logger actually writes messages."""
        log_dir = tmp_path / "logs"
        logger = setup_logging(str(log_dir), "test")

        test_message = "Test log message"
        logger.info(test_message)

        log_file = log_dir / "test.log"
        content = log_file.read_text()
        assert test_message in content

    def test_setup_logging_different_names(self, tmp_path: Path) -> None:
        """Test creating loggers with different names."""
        log_dir = tmp_path / "logs"
        logger1 = setup_logging(str(log_dir), "logger1")
        logger2 = setup_logging(str(log_dir), "logger2")

        assert logger1.name == "logger1"
        assert logger2.name == "logger2"

        # Both should have created log files
        assert (log_dir / "logger1.log").exists()
        assert (log_dir / "logger2.log").exists()


class TestSetGlobalSeed:
    """Tests for set_global_seed function."""

    def test_set_global_seed_reproducibility(self) -> None:
        """Test that same seed produces same random numbers."""
        set_global_seed(42)
        vals1 = np.random.randn(100)

        set_global_seed(42)
        vals2 = np.random.randn(100)

        assert np.allclose(vals1, vals2)

    def test_set_global_seed_different_values(self) -> None:
        """Test that different seeds produce different random values."""
        set_global_seed(42)
        vals1 = np.random.randn(100)

        set_global_seed(99)
        vals2 = np.random.randn(100)

        # Should not be identical (very low probability)
        assert not np.allclose(vals1, vals2)

    def test_set_global_seed_affects_numpy(self) -> None:
        """Test that seed affects numpy random generation."""
        set_global_seed(100)
        arr1 = np.random.rand(5, 5)

        set_global_seed(100)
        arr2 = np.random.rand(5, 5)

        assert np.allclose(arr1, arr2)


class TestRunBasename:
    """Tests for run_basename function."""

    def test_run_basename_subject_only(self) -> None:
        """Test basename with only subject."""
        row = {"subject": "001"}
        result = run_basename(row)
        assert result == "sub-001"

    def test_run_basename_with_session(self) -> None:
        """Test basename with subject and session."""
        row = {"subject": "001", "session": "01"}
        result = run_basename(row)
        assert result == "sub-001_ses-01"

    def test_run_basename_with_task(self) -> None:
        """Test basename with subject and task."""
        row = {"subject": "001", "task": "rest"}
        result = run_basename(row)
        assert result == "sub-001_task-rest"

    def test_run_basename_with_run(self) -> None:
        """Test basename with subject, task, and run."""
        row = {"subject": "001", "task": "rest", "run": "01"}
        result = run_basename(row)
        assert result == "sub-001_task-rest_run-01"

    def test_run_basename_full(self) -> None:
        """Test basename with all BIDS entities."""
        row = {
            "subject": "001",
            "session": "01",
            "task": "rest",
            "run": "02",
        }
        result = run_basename(row)
        assert result == "sub-001_ses-01_task-rest_run-02"

    def test_run_basename_missing_keys(self) -> None:
        """Test basename handles missing keys gracefully."""
        row = {"subject": "001"}  # Missing session, task, run
        result = run_basename(row)
        # Should still create valid basename
        assert result.startswith("sub-001")
        assert "None" not in result

    def test_run_basename_session_without_task(self) -> None:
        """Test ordering: subject -> session -> task -> run."""
        row = {"subject": "001", "session": "01", "run": "01"}
        result = run_basename(row)
        # Task should not appear if missing
        assert result == "sub-001_ses-01_run-01"


class TestMetricPath:
    """Tests for metric_path standardized path construction."""

    def test_metric_path_basic(self, tmp_path: Path) -> None:
        """Test basic metric path construction."""
        path = metric_path(str(tmp_path), "connectivity", "001")

        assert str(tmp_path) in str(path)
        assert "connectivity" in str(path)
        assert "sub-001" in str(path)

    def test_metric_path_creates_directory(self, tmp_path: Path) -> None:
        """Test that metric_path creates the directory."""
        path = metric_path(str(tmp_path), "reho", "002")

        assert path.exists()
        assert path.is_dir()

    def test_metric_path_with_run_key(self, tmp_path: Path) -> None:
        """Test metric_path with run key."""
        path = metric_path(str(tmp_path), "roi", "001", "sub-001_task-rest")

        assert "sub-001_task-rest" in str(path)

    def test_metric_path_nested_creation(self, tmp_path: Path) -> None:
        """Test that nested directories are created."""
        path = metric_path(str(tmp_path), "pca/metrics", "005", "sub-005_ses-02")

        assert path.exists()

    def test_metric_path_multiple_metrics(self, tmp_path: Path) -> None:
        """Test creating paths for different metrics."""
        paths = [
            metric_path(str(tmp_path), "connectivity", "001"),
            metric_path(str(tmp_path), "reho", "001"),
            metric_path(str(tmp_path), "roi", "001"),
        ]

        assert all(p.exists() for p in paths)
        assert len(set(str(p) for p in paths)) == 3  # All unique


class TestSaveJson:
    """Tests for save_json serialization."""

    def test_save_json_basic(self, tmp_path: Path) -> None:
        """Test saving basic dict to JSON."""
        obj = {"key": "value", "number": 42}
        path = tmp_path / "test.json"

        save_json(obj, str(path))

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded == obj

    def test_save_json_nested(self, tmp_path: Path) -> None:
        """Test saving nested dictionary."""
        obj = {
            "level1": {
                "level2": {
                    "list": [1, 2, 3],
                    "string": "nested",
                }
            }
        }
        path = tmp_path / "nested.json"

        save_json(obj, str(path))

        loaded = json.loads(path.read_text())
        assert loaded == obj

    def test_save_json_creates_directory(self, tmp_path: Path) -> None:
        """Test that save_json creates parent directories."""
        obj = {"test": "data"}
        nested_path = tmp_path / "a" / "b" / "c" / "test.json"

        save_json(obj, str(nested_path))

        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_save_json_overwrites(self, tmp_path: Path) -> None:
        """Test that save_json overwrites existing files."""
        path = tmp_path / "test.json"

        save_json({"version": 1}, str(path))
        save_json({"version": 2}, str(path))

        loaded = json.loads(path.read_text())
        assert loaded["version"] == 2

    def test_save_json_numeric_types(self, tmp_path: Path) -> None:
        """Test JSON serialization of numeric types."""
        obj = {
            "int": 42,
            "float": 3.14,
            "list": [1, 2.5, 3],
            "numpy_int": int(np.int32(100)),
            "numpy_float": float(np.float64(2.718)),
        }
        path = tmp_path / "numbers.json"

        save_json(obj, str(path))

        loaded = json.loads(path.read_text())
        assert loaded["int"] == 42
        assert loaded["float"] == pytest.approx(3.14)


class TestUpperTriangleVector:
    """Tests for upper_triangle_vector extraction."""

    def test_upper_triangle_vector_basic(self) -> None:
        """Test extracting upper triangle from square matrix."""
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)

        vec, idx = upper_triangle_vector(mat)

        # Upper triangle (k=1): 2, 3, 6
        expected = np.array([2, 3, 6], dtype=float)
        assert np.allclose(vec, expected)

    def test_upper_triangle_vector_indices(self) -> None:
        """Test that returned indices match matrix."""
        mat = np.random.randn(5, 5)

        vec, idx = upper_triangle_vector(mat)

        # Verify indices match matrix values
        i, j = idx
        mat_vals = mat[i, j]
        assert np.allclose(vec, mat_vals)

    def test_upper_triangle_vector_different_k(self) -> None:
        """Test extracting with different diagonal offsets."""
        mat = np.arange(25, dtype=float).reshape(5, 5)

        # k=0 includes main diagonal
        vec_k0, idx_k0 = upper_triangle_vector(mat, k=0)
        assert len(vec_k0) > 0

        # k=1 excludes main diagonal (default)
        vec_k1, idx_k1 = upper_triangle_vector(mat, k=1)
        assert len(vec_k1) > 0

        # k=1 should have fewer elements
        assert len(vec_k1) < len(vec_k0)

    def test_upper_triangle_vector_shape(self) -> None:
        """Test output shapes for various matrix sizes."""
        for n in [3, 5, 10]:
            mat = np.random.randn(n, n)
            vec, idx = upper_triangle_vector(mat)

            # Upper triangle count: n*(n-1)/2
            expected_count = n * (n - 1) // 2
            assert len(vec) == expected_count

    def test_upper_triangle_vector_symmetry(self) -> None:
        """Test upper_triangle_vector on symmetric matrix."""
        mat = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

        vec, idx = upper_triangle_vector(mat)
        expected = np.array([2, 3, 5])

        assert np.allclose(vec, expected)

    def test_upper_triangle_vector_ones(self) -> None:
        """Test on all-ones matrix."""
        mat = np.ones((4, 4))

        vec, idx = upper_triangle_vector(mat)

        # All elements should be 1.0
        assert np.allclose(vec, 1.0)

    def test_upper_triangle_vector_returntype(self) -> None:
        """Test that return types are correct."""
        mat = np.random.randn(5, 5)

        vec, idx = upper_triangle_vector(mat)

        assert isinstance(vec, np.ndarray)
        assert isinstance(idx, tuple)
        assert len(idx) == 2  # Two index arrays (row, col)
