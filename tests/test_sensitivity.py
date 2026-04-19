from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add src to path for imports, and scripts
SRC_DIR = Path(__file__).parent.parent / "src"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from run_sensitivity_analysis import (
    BASELINE,
    ParameterSet,
    build_cfg_for_params,
    build_parameter_grid,
)


class TestParameterSet:
    """Tests for ParameterSet class."""

    def test_parameter_set_label(self) -> None:
        """Test ParameterSet.label() format."""
        params = ParameterSet(
            name="test",
            gsr=True,
            schaefer_n_rois=200,
            dfc_window_trs=30,
            fd_threshold_mm=0.5,
            smoothing_fwhm_mm=6.0,
        )
        label = params.label()

        assert "GSR=on" in label
        assert "atlas=200" in label
        assert "dfc=30TR" in label
        assert "FD=0.5" in label
        assert "sm=6.0mm" in label

    def test_parameter_set_label_gsr_off(self) -> None:
        """Test label with GSR off."""
        params = ParameterSet(name="test_gsr_off", gsr=False)
        label = params.label()
        assert "GSR=off" in label

    def test_parameter_set_label_different_atlas(self) -> None:
        """Test label with different atlas size."""
        params = ParameterSet(name="test_atlas", schaefer_n_rois=400)
        label = params.label()
        assert "atlas=400" in label

    def test_parameter_set_to_dict(self) -> None:
        """Test conversion to dictionary."""
        params = ParameterSet(
            name="test_dict",
            gsr=True,
            schaefer_n_rois=200,
            dfc_window_trs=30,
            fd_threshold_mm=0.5,
            smoothing_fwhm_mm=6.0,
        )
        d = params.to_dict()

        assert d["name"] == "test_dict"
        assert d["gsr"] is True
        assert d["schaefer_n_rois"] == 200
        assert d["dfc_window_trs"] == 30
        assert d["fd_threshold_mm"] == 0.5
        assert d["smoothing_fwhm_mm"] == 6.0


class TestBuildParameterGrid:
    """Tests for build_parameter_grid function."""

    def test_build_parameter_grid_count(self) -> None:
        """Test grid has expected number of conditions.

        Grid should have:
        1 baseline
        + 1 GSR off
        + 2 atlas variations (100, 400)
        + 3 dFC window variations (20, 45, 60)
        + 2 scrubbing variations (0.3, 0.9)
        + 2 smoothing variations (4.0, 8.0)
        = 11 total
        """
        grid = build_parameter_grid()
        assert len(grid) == 11

    def test_build_parameter_grid_baseline_present(self) -> None:
        """Test that baseline is first in grid."""
        grid = build_parameter_grid()
        assert grid[0].name == "baseline"

    def test_build_parameter_grid_baseline_matches(self) -> None:
        """Test baseline parameters match BASELINE constant."""
        grid = build_parameter_grid()
        baseline = grid[0]

        assert baseline.gsr == BASELINE.gsr
        assert baseline.schaefer_n_rois == BASELINE.schaefer_n_rois
        assert baseline.dfc_window_trs == BASELINE.dfc_window_trs
        assert baseline.fd_threshold_mm == BASELINE.fd_threshold_mm
        assert baseline.smoothing_fwhm_mm == BASELINE.smoothing_fwhm_mm

    def test_build_parameter_grid_gsr_variation(self) -> None:
        """Test GSR variation in grid."""
        grid = build_parameter_grid()
        gsr_off = [p for p in grid if p.name == "gsr_off"]

        assert len(gsr_off) == 1
        assert gsr_off[0].gsr is False
        assert gsr_off[0].schaefer_n_rois == BASELINE.schaefer_n_rois

    def test_build_parameter_grid_atlas_variations(self) -> None:
        """Test atlas variations in grid."""
        grid = build_parameter_grid()
        atlas_names = [p.name for p in grid if p.name.startswith("atlas_")]

        assert "atlas_100" in atlas_names
        assert "atlas_400" in atlas_names
        assert len(atlas_names) == 2

    def test_build_parameter_grid_dfc_variations(self) -> None:
        """Test dFC window variations in grid."""
        grid = build_parameter_grid()
        dfc_names = [p.name for p in grid if p.name.startswith("dfc_")]

        assert "dfc_20TR" in dfc_names
        assert "dfc_45TR" in dfc_names
        assert "dfc_60TR" in dfc_names
        assert len(dfc_names) == 3

    def test_build_parameter_grid_scrubbing_variations(self) -> None:
        """Test scrubbing threshold variations in grid."""
        grid = build_parameter_grid()
        scrub_names = [p.name for p in grid if p.name.startswith("FD_")]

        assert "FD_0.3mm" in scrub_names
        assert "FD_0.9mm" in scrub_names
        assert len(scrub_names) == 2

    def test_build_parameter_grid_smoothing_variations(self) -> None:
        """Test smoothing variations in grid."""
        grid = build_parameter_grid()
        smooth_names = [p.name for p in grid if p.name.startswith("smooth_")]

        assert "smooth_4.0mm" in smooth_names
        assert "smooth_8.0mm" in smooth_names
        assert len(smooth_names) == 2

    def test_build_parameter_grid_unique_names(self) -> None:
        """Test that all parameter sets have unique names."""
        grid = build_parameter_grid()
        names = [p.name for p in grid]
        assert len(names) == len(set(names))


class TestBuildCfgForParams:
    """Tests for build_cfg_for_params function."""

    def test_build_cfg_for_params_structure(self, tmp_path) -> None:
        """Test config dict structure."""
        params = BASELINE
        cfg = build_cfg_for_params(params, tmp_path)

        assert "project" in cfg
        assert "paths" in cfg
        assert "preprocessing" in cfg
        assert "roi" in cfg
        assert "dynamic_fc" in cfg
        assert "reho" in cfg

    def test_build_cfg_for_params_project_section(self, tmp_path) -> None:
        """Test project section of config."""
        params = BASELINE
        cfg = build_cfg_for_params(params, tmp_path)

        assert "name" in cfg["project"]
        assert "random_seed" in cfg["project"]
        assert "n_jobs" in cfg["project"]
        assert cfg["project"]["random_seed"] == 42

    def test_build_cfg_for_params_paths_section(self, tmp_path) -> None:
        """Test paths section of config."""
        params = BASELINE
        cfg = build_cfg_for_params(params, tmp_path)

        assert "output_root" in cfg["paths"]
        assert "cache_dir" in cfg["paths"]
        assert "logs_dir" in cfg["paths"]
        # Should contain the output root path
        assert str(tmp_path) in cfg["paths"]["output_root"]

    def test_build_cfg_for_params_preprocessing_gsr(self, tmp_path) -> None:
        """Test preprocessing section respects GSR parameter."""
        params_on = ParameterSet(name="test_gsr_on", gsr=True)
        cfg_on = build_cfg_for_params(params_on, tmp_path)
        assert cfg_on["preprocessing"]["confounds"]["include_global_signal"] is True

        params_off = ParameterSet(name="test_gsr_off", gsr=False)
        cfg_off = build_cfg_for_params(params_off, tmp_path)
        assert cfg_off["preprocessing"]["confounds"]["include_global_signal"] is False

    def test_build_cfg_for_params_preprocessing_fd(self, tmp_path) -> None:
        """Test preprocessing section has FD threshold."""
        params = ParameterSet(name="test", fd_threshold_mm=0.7)
        cfg = build_cfg_for_params(params, tmp_path)

        assert cfg["preprocessing"]["scrubbing"]["fd_threshold_mm"] == 0.7

    def test_build_cfg_for_params_preprocessing_smoothing(self, tmp_path) -> None:
        """Test preprocessing smoothing kernel."""
        params = ParameterSet(name="test", smoothing_fwhm_mm=8.0)
        cfg = build_cfg_for_params(params, tmp_path)

        assert cfg["preprocessing"]["smoothing_fwhm_mm"] == 8.0

    def test_build_cfg_for_params_roi_atlas(self, tmp_path) -> None:
        """Test ROI section atlas specification."""
        params = ParameterSet(name="test", schaefer_n_rois=400)
        cfg = build_cfg_for_params(params, tmp_path)

        assert cfg["roi"]["atlas"] == "schaefer_400"
        assert cfg["roi"]["schaefer_n_rois"] == 400

    def test_build_cfg_for_params_dynamic_fc(self, tmp_path) -> None:
        """Test dynamic FC section."""
        params = ParameterSet(name="test", dfc_window_trs=45)
        cfg = build_cfg_for_params(params, tmp_path)

        assert cfg["dynamic_fc"]["window_trs"] == 45
        assert cfg["dynamic_fc"]["step_trs"] == 5  # Default

    def test_build_cfg_for_params_reho_section(self, tmp_path) -> None:
        """Test ReHo section is present."""
        params = BASELINE
        cfg = build_cfg_for_params(params, tmp_path)

        assert cfg["reho"]["neighborhood"] == 26
        assert cfg["reho"]["normalize_zscore"] is True
        assert cfg["reho"]["chunk_size_voxels"] == 10000

    def test_build_cfg_for_params_project_name_includes_param_name(self, tmp_path) -> None:
        """Test that project name includes parameter set name."""
        params = ParameterSet(name="special_test")
        cfg = build_cfg_for_params(params, tmp_path)

        assert "special_test" in cfg["project"]["name"]
