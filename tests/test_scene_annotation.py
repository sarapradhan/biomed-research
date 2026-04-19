from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path for imports
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from fmri_pipeline.scene_annotation import (
    align_isc_to_scenes,
    correlate_isc_with_features,
    create_annotation_template,
    load_scene_annotations,
    scenes_to_tr_indices,
)


class TestLoadSceneAnnotations:
    """Tests for load_scene_annotations function."""

    def test_load_scene_annotations_valid(self, tmp_path) -> None:
        """Test loading valid scene annotation CSV."""
        csv_path = tmp_path / "scenes.csv"
        df = pd.DataFrame({
            "onset_sec": [0.0, 45.0, 90.0],
            "offset_sec": [45.0, 90.0, 135.0],
            "emotional_valence": [0.5, -0.3, 0.8],
            "social_cognition": [0, 1, 0],
            "narrative_transition": [0, 1, 0],
        })
        df.to_csv(csv_path, index=False)

        result = load_scene_annotations(str(csv_path))
        assert len(result) == 3
        assert "onset_sec" in result.columns
        assert "offset_sec" in result.columns
        assert "emotional_valence" in result.columns

    def test_load_scene_annotations_missing_timing_columns(self, tmp_path) -> None:
        """Test ValueError raised for missing timing columns."""
        csv_path = tmp_path / "scenes.csv"
        df = pd.DataFrame({
            "emotional_valence": [0.5, -0.3],
            "social_cognition": [0, 1],
        })
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required column"):
            load_scene_annotations(str(csv_path))

    def test_load_scene_annotations_missing_feature_columns(self, tmp_path) -> None:
        """Test ValueError raised for missing all feature columns."""
        csv_path = tmp_path / "scenes.csv"
        df = pd.DataFrame({
            "onset_sec": [0.0, 45.0],
            "offset_sec": [45.0, 90.0],
        })
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="No feature columns found"):
            load_scene_annotations(str(csv_path))

    def test_load_scene_annotations_invalid_timing(self, tmp_path) -> None:
        """Test ValueError for invalid timing (offset <= onset)."""
        csv_path = tmp_path / "scenes.csv"
        df = pd.DataFrame({
            "onset_sec": [0.0, 90.0],
            "offset_sec": [45.0, 90.0],  # Second row has offset == onset
            "emotional_valence": [0.5, 0.3],
        })
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Invalid timing"):
            load_scene_annotations(str(csv_path))

    def test_load_scene_annotations_custom_required_columns(self, tmp_path) -> None:
        """Test loading with custom required columns."""
        csv_path = tmp_path / "scenes.csv"
        df = pd.DataFrame({
            "onset_sec": [0.0, 45.0],
            "offset_sec": [45.0, 90.0],
            "custom_feature": [1.0, 2.0],
        })
        df.to_csv(csv_path, index=False)

        result = load_scene_annotations(str(csv_path), required_columns=["custom_feature"])
        assert "custom_feature" in result.columns
        assert len(result) == 2


class TestScenesToTRIndices:
    """Tests for scenes_to_tr_indices function."""

    def test_scenes_to_tr_indices_basic(self) -> None:
        """Test basic conversion from seconds to TR indices."""
        scenes_df = pd.DataFrame({
            "onset_sec": [0.0, 45.0],
            "offset_sec": [45.0, 90.0],
        })
        tr_sec = 2.0
        result = scenes_to_tr_indices(scenes_df, tr_sec)

        assert "onset_tr" in result.columns
        assert "offset_tr" in result.columns
        assert "duration_trs" in result.columns
        assert result["onset_tr"].iloc[0] == 0
        assert result["onset_tr"].iloc[1] == 22  # floor(45/2)
        assert result["offset_tr"].iloc[0] == 23  # ceil(45/2)
        assert result["offset_tr"].iloc[1] == 45  # ceil(90/2)

    def test_scenes_to_tr_indices_clipping(self) -> None:
        """Test that offset_tr is clipped to n_volumes."""
        scenes_df = pd.DataFrame({
            "onset_sec": [0.0, 45.0],
            "offset_sec": [45.0, 250.0],  # 250/2=125 TRs > n_volumes=100, clips to 100
        })
        tr_sec = 2.0
        n_volumes = 100

        result = scenes_to_tr_indices(scenes_df, tr_sec, n_volumes)
        # 250 / 2 = 125, ceil(125) = 125, clipped to 100
        assert result["offset_tr"].iloc[1] == 100

    def test_scenes_to_tr_indices_duration(self) -> None:
        """Test duration_trs calculation."""
        scenes_df = pd.DataFrame({
            "onset_sec": [0.0, 30.0],
            "offset_sec": [30.0, 60.0],
        })
        tr_sec = 2.0

        result = scenes_to_tr_indices(scenes_df, tr_sec)
        # First scene: 15 - 0 = 15 TRs
        # Second scene: 30 - 15 = 15 TRs
        assert result["duration_trs"].iloc[0] == 15
        assert result["duration_trs"].iloc[1] == 15

    def test_scenes_to_tr_indices_preserves_other_columns(self) -> None:
        """Test that other columns are preserved."""
        scenes_df = pd.DataFrame({
            "onset_sec": [0.0],
            "offset_sec": [45.0],
            "emotional_valence": [0.5],
            "my_feature": ["test"],
        })
        tr_sec = 2.0

        result = scenes_to_tr_indices(scenes_df, tr_sec)
        assert "emotional_valence" in result.columns
        assert "my_feature" in result.columns
        assert result["emotional_valence"].iloc[0] == 0.5
        assert result["my_feature"].iloc[0] == "test"


class TestAlignISCToScenes:
    """Tests for align_isc_to_scenes function."""

    def test_align_isc_to_scenes_basic(self) -> None:
        """Test basic alignment of ISC to scenes."""
        isc_timecourse = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        scenes_df = pd.DataFrame({
            "onset_tr": [0, 5],
            "offset_tr": [5, 10],
        })

        result = align_isc_to_scenes(isc_timecourse, scenes_df, summary_method="mean", zscore_isc=False)

        assert "scene_isc" in result.columns
        # First scene: mean of [0.1, 0.2, 0.3, 0.4, 0.5] = 0.3
        assert np.isclose(result["scene_isc"].iloc[0], 0.3)
        # Second scene: mean of [0.6, 0.7, 0.8, 0.9, 1.0] = 0.8
        assert np.isclose(result["scene_isc"].iloc[1], 0.8)

    def test_align_isc_to_scenes_median(self) -> None:
        """Test with median summary method."""
        isc_timecourse = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        scenes_df = pd.DataFrame({
            "onset_tr": [0, 5],
            "offset_tr": [5, 10],
        })

        result = align_isc_to_scenes(isc_timecourse, scenes_df, summary_method="median", zscore_isc=False)

        # First scene median: median of [1, 2, 3, 4, 5] = 3
        assert np.isclose(result["scene_isc"].iloc[0], 3.0)
        # Second scene median: median of [6, 7, 8, 9, 10] = 8
        assert np.isclose(result["scene_isc"].iloc[1], 8.0)

    def test_align_isc_to_scenes_zscore(self) -> None:
        """Test z-scoring of ISC."""
        isc_timecourse = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        scenes_df = pd.DataFrame({
            "onset_tr": [0, 5],
            "offset_tr": [5, 10],
        })

        result = align_isc_to_scenes(isc_timecourse, scenes_df, zscore_isc=True)

        # With z-scoring, mean should be ~0, std ~1 across all values
        all_isc = result["scene_isc"].dropna()
        assert np.isclose(np.mean(all_isc), 0.0, atol=0.5)

    def test_align_isc_to_scenes_short_scene(self) -> None:
        """Test scenes shorter than min_scene_trs."""
        isc_timecourse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scenes_df = pd.DataFrame({
            "onset_tr": [0, 3],
            "offset_tr": [1, 4],  # Both are 1 TR, less than default min_scene_trs=5
        })

        result = align_isc_to_scenes(isc_timecourse, scenes_df, min_scene_trs=5)

        assert np.isnan(result["scene_isc"].iloc[0])
        assert np.isnan(result["scene_isc"].iloc[1])

    def test_align_isc_to_scenes_out_of_bounds(self) -> None:
        """Test scenes that extend beyond ISC timecourse."""
        isc_timecourse = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scenes_df = pd.DataFrame({
            "onset_tr": [2, 10],
            "offset_tr": [5, 15],  # Second scene completely out of bounds
        })

        result = align_isc_to_scenes(isc_timecourse, scenes_df, min_scene_trs=1)

        assert not np.isnan(result["scene_isc"].iloc[0])  # Valid
        assert np.isnan(result["scene_isc"].iloc[1])  # Out of bounds


class TestCorrelateISCWithFeatures:
    """Tests for correlate_isc_with_features function."""

    def test_correlate_isc_with_features_basic(self) -> None:
        """Test basic correlation of ISC with features."""
        aligned_df = pd.DataFrame({
            "scene_isc": [0.1, 0.2, 0.3, 0.4, 0.5],
            "emotional_valence": [0.0, 0.25, 0.5, 0.75, 1.0],
        })

        result = correlate_isc_with_features(aligned_df, ["emotional_valence"], method="spearman")

        assert len(result) == 1
        assert result["feature"].iloc[0] == "emotional_valence"
        assert "r" in result.columns
        assert "p_value" in result.columns
        assert result["n_scenes"].iloc[0] == 5

    def test_correlate_isc_with_features_multiple(self) -> None:
        """Test correlation with multiple features."""
        aligned_df = pd.DataFrame({
            "scene_isc": [0.1, 0.2, 0.3, 0.4, 0.5],
            "emotional_valence": [0.0, 0.25, 0.5, 0.75, 1.0],
            "social_cognition": [0, 1, 0, 1, 0],
        })

        result = correlate_isc_with_features(aligned_df, ["emotional_valence", "social_cognition"])

        assert len(result) == 2
        assert "emotional_valence" in result["feature"].values
        assert "social_cognition" in result["feature"].values

    def test_correlate_isc_with_features_missing_column(self) -> None:
        """Test behavior with missing feature column."""
        aligned_df = pd.DataFrame({
            "scene_isc": [0.1, 0.2, 0.3],
            "emotional_valence": [0.0, 0.5, 1.0],
        })

        result = correlate_isc_with_features(aligned_df, ["nonexistent_feature"])

        assert len(result) == 0

    def test_correlate_isc_with_features_no_variance(self) -> None:
        """Test behavior when feature has no variance."""
        aligned_df = pd.DataFrame({
            "scene_isc": [0.1, 0.2, 0.3],
            "constant_feature": [1.0, 1.0, 1.0],  # No variance
        })

        result = correlate_isc_with_features(aligned_df, ["constant_feature"])

        assert len(result) == 1
        assert np.isnan(result["r"].iloc[0])

    def test_correlate_isc_with_features_nan_values(self) -> None:
        """Test behavior with NaN values."""
        aligned_df = pd.DataFrame({
            "scene_isc": [0.1, np.nan, 0.3, 0.4],
            "emotional_valence": [0.0, 0.25, 0.5, 0.75],
        })

        result = correlate_isc_with_features(aligned_df, ["emotional_valence"])

        # Should only use valid rows (3 scenes)
        assert result["n_scenes"].iloc[0] == 3

    def test_correlate_isc_with_features_pearson(self) -> None:
        """Test Pearson correlation method."""
        aligned_df = pd.DataFrame({
            "scene_isc": [0.1, 0.2, 0.3, 0.4],
            "feature": [1.0, 2.0, 3.0, 4.0],
        })

        result = correlate_isc_with_features(aligned_df, ["feature"], method="pearson")

        assert result["method"].iloc[0] == "pearson"
        assert result["r"].iloc[0] > 0.9  # Should be high correlation


class TestCreateAnnotationTemplate:
    """Tests for create_annotation_template function."""

    def test_create_annotation_template_default(self, tmp_path) -> None:
        """Test creating template with default features."""
        output_path = str(tmp_path / "template.csv")
        result_path = create_annotation_template(output_path)

        assert Path(result_path).exists()
        df = pd.read_csv(result_path)

        # Check required columns
        assert "onset_sec" in df.columns
        assert "offset_sec" in df.columns
        assert "emotional_valence" in df.columns
        assert "social_cognition" in df.columns
        assert "narrative_transition" in df.columns

    def test_create_annotation_template_custom_features(self, tmp_path) -> None:
        """Test creating template with custom features."""
        output_path = str(tmp_path / "template.csv")
        result_path = create_annotation_template(
            output_path,
            features=["custom_feature_1", "custom_feature_2"]
        )

        df = pd.read_csv(result_path)
        assert "custom_feature_1" in df.columns
        assert "custom_feature_2" in df.columns
        assert "emotional_valence" not in df.columns

    def test_create_annotation_template_rows(self, tmp_path) -> None:
        """Test creating template with specific number of rows."""
        output_path = str(tmp_path / "template.csv")
        n_rows = 10
        result_path = create_annotation_template(output_path, n_example_rows=n_rows)

        df = pd.read_csv(result_path)
        assert len(df) == n_rows

    def test_create_annotation_template_timing(self, tmp_path) -> None:
        """Test that template has reasonable timing."""
        output_path = str(tmp_path / "template.csv")
        result_path = create_annotation_template(output_path, n_example_rows=3)

        df = pd.read_csv(result_path)
        # Should be sorted by onset
        assert df["onset_sec"].is_monotonic_increasing
        # All offsets should be > onsets
        assert (df["offset_sec"] > df["onset_sec"]).all()
        # Rows should be non-overlapping (offset of row i <= onset of row i+1)
        for i in range(len(df) - 1):
            assert df["offset_sec"].iloc[i] <= df["onset_sec"].iloc[i + 1]
