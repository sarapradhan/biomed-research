# Scene Annotation Configuration Schema

## Overview

The `scene_annotation` configuration block enables the Track 2 ISC extension, which aligns intersubject correlation (ISC) maps with scene-level annotations of stimulus content features (emotional valence, social cognition, narrative structure, etc.).

This document specifies the complete schema for `scene_annotation` configuration in YAML.

---

## Configuration Block: `scene_annotation`

### Top-Level Structure

```yaml
scene_annotation:
  enabled: bool
  annotation_dir: str
  tr_sec: float
  features: list[str]
  alignment: dict
  correlation_method: str (optional)
```

### Field Definitions

#### `enabled` (Required)
- **Type:** `bool`
- **Default:** `false`
- **Description:** Gate to enable/disable the scene alignment step. If `false`, `run_scene_alignment_step()` returns `{}` without processing.
- **Example:**
  ```yaml
  enabled: true
  ```

#### `annotation_dir` (Required if `enabled: true`)
- **Type:** `str` (file path)
- **Description:** Absolute or relative path to directory containing scene annotation CSV files. The pipeline will create this directory if it doesn't exist.
  - One CSV file per episode or run
  - File naming convention: arbitrary (e.g., `episode_01.csv`, `run_friends_s01e01_scenes.csv`)
  - All CSVs must contain `onset_sec` and `offset_sec` columns
  - All CSVs must contain at least one feature column matching those listed in `features`
- **Example:**
  ```yaml
  annotation_dir: /path/to/output/cneuromod_isc/annotations
  ```

#### `tr_sec` (Required if `enabled: true`)
- **Type:** `float`
- **Range:** Positive real number (typically 1.0–3.0 for fMRI)
- **Description:** Repetition time in seconds for converting scene onset/offset times (in seconds) to volume indices (TRs). Must match the TR of the preprocessed fMRI data.
- **Example:**
  ```yaml
  tr_sec: 1.49   # CNeuroMod Friends TR
  ```

#### `features` (Required if `enabled: true`)
- **Type:** `list[str]`
- **Description:** List of scene feature column names to include in analysis. These columns must be present in all annotation CSVs. Recommended features:
  - `emotional_valence`: float in range [-1, +1] (-1 = negative, 0 = neutral, +1 = positive)
  - `social_cognition`: int (0 = no theory-of-mind, 1 = theory-of-mind moment)
  - `narrative_transition`: int (0 = within-scene continuity, 1 = scene boundary)
- Custom features are allowed (any float or int column).
- **Example:**
  ```yaml
  features:
    - emotional_valence
    - social_cognition
    - narrative_transition
  ```

#### `alignment` (Optional)
- **Type:** `dict`
- **Description:** Subblock controlling ISC-to-scene alignment method.
- **Default:** `{}`
- **Subfields:**
  
  ##### `alignment.summary_method`
  - **Type:** `str` (enum: `"mean"` or `"median"`)
  - **Default:** `"mean"`
  - **Description:** Method for summarizing ISC values within each scene segment.
  - **Example:**
    ```yaml
    alignment:
      summary_method: mean
    ```
  
  ##### `alignment.min_scene_trs`
  - **Type:** `int`
  - **Default:** `5`
  - **Range:** Positive integer
  - **Description:** Minimum scene duration (in TRs) to include in alignment. Scenes shorter than this are excluded. Use to avoid aliasing from very brief scenes.
  - **Example:**
    ```yaml
    alignment:
      min_scene_trs: 5
    ```
  
  ##### `alignment.zscore_isc`
  - **Type:** `bool`
  - **Default:** `true`
  - **Description:** Whether to z-score ISC values before correlating with scene features. Recommended to standardize scale and improve interpretability.
  - **Example:**
    ```yaml
    alignment:
      zscore_isc: true
    ```

#### `correlation_method` (Optional)
- **Type:** `str` (enum: `"pearson"` or `"spearman"`)
- **Default:** `"pearson"`
- **Description:** Correlation method for relating ISC values to scene features.
- **Example:**
  ```yaml
  correlation_method: pearson
  ```

---

## Annotation CSV Format

Scene annotations are stored in plain-text CSV files with the following required structure:

### Required Columns
- `onset_sec` (float): Scene start time in seconds from stimulus onset
- `offset_sec` (float): Scene end time in seconds from stimulus onset

### Feature Columns
At least one of the columns listed in `scene_annotation.features` must be present. Common features:

| Column Name | Type | Range | Interpretation |
|---|---|---|---|
| `emotional_valence` | float | [-1, +1] | -1 (negative), 0 (neutral), +1 (positive) |
| `social_cognition` | int | {0, 1} | 0 (no ToM), 1 (theory-of-mind) |
| `narrative_transition` | int | {0, 1} | 0 (within-scene), 1 (scene boundary) |

### Example Annotation CSV

```csv
onset_sec,offset_sec,emotional_valence,social_cognition,narrative_transition
0.0,45.2,0.3,0,0
45.2,78.6,-0.5,1,1
78.6,120.0,0.8,0,0
120.0,165.3,0.1,1,0
```

### Validation Rules
1. ✅ `onset_sec < offset_sec` for all rows
2. ✅ All required feature columns present
3. ✅ Feature values within documented ranges
4. ✅ Scenes should not overlap (soft check; overlaps are allowed but may lead to ambiguity)
5. ✅ CSV is sorted by `onset_sec`

---

## Complete Configuration Example

### Minimal Valid Config
```yaml
scene_annotation:
  enabled: true
  annotation_dir: /path/to/annotations
  tr_sec: 1.5
  features:
    - emotional_valence
```

### Recommended Config (CNeuroMod Friends)
```yaml
scene_annotation:
  enabled: true
  annotation_dir: /path/to/output/cneuromod_isc/annotations
  tr_sec: 1.49
  features:
    - emotional_valence
    - social_cognition
    - narrative_transition
  alignment:
    summary_method: mean
    min_scene_trs: 5
    zscore_isc: true
  correlation_method: pearson
```

### Disabled Config
```yaml
scene_annotation:
  enabled: false
```

---

## Validation & Error Handling

### Pre-Flight Checks
The pipeline performs the following checks **before** running scene alignment:

1. **Config Presence**: If `scene_annotation` key is missing, defaults to `enabled: false`
2. **Enable Gate**: If `enabled: false`, step is skipped entirely
3. **Directory Creation**: `annotation_dir` is created if it doesn't exist
4. **Annotation Loading**: CSVs are loaded from `annotation_dir/*.csv`
   - If no CSVs found, a template is generated and user is prompted to fill it in
   - If CSVs found but missing required columns, those CSVs are skipped with warnings

### Runtime Errors & Handling

| Condition | Behavior | Log Level |
|---|---|---|
| `enabled: false` | Step skipped | info |
| No annotations found | Template created, user prompted | info |
| Missing feature column in CSV | CSV skipped | warning |
| Invalid `tr_sec` value | Raises `ValueError` | error |
| `n_volumes` not determinable | Uses default (200) with warning | warning |
| Scene onset > stimulus duration | Scene clipped or excluded | warning |
| `alignment.summary_method` invalid | Uses default ("mean") | warning |

---

## TR Index Conversion

The pipeline converts scene timing (seconds) to TR indices as follows:

```
start_tr = round(onset_sec / tr_sec)
end_tr = round(offset_sec / tr_sec)
```

Example:
- `onset_sec = 45.2`, `offset_sec = 78.6`, `tr_sec = 1.49`
- `start_tr = round(45.2 / 1.49) = 30`
- `end_tr = round(78.6 / 1.49) = 53`

---

## Output Artifacts

When `scene_annotation.enabled: true` and annotations are loaded, the following outputs are generated:

### Directory Structure
```
output_root/
  scene_analysis/
    {run_name}/
      isc_scene_alignment.csv
      isc_feature_correlations.csv
      isc_vs_emotional_valence.png
      isc_vs_social_cognition.png
      isc_vs_narrative_transition.png
      (+ one plot per feature column)
```

### Output Files

#### `isc_scene_alignment.csv`
- **Columns:** `scene_id`, `start_tr`, `end_tr`, (feature columns), `isc_mean` (or `isc_median`)
- **Rows:** One per annotated scene (after filtering by `min_scene_trs`)
- **Description:** Aligned ISC values and scene features

#### `isc_feature_correlations.csv`
- **Columns:** `feature`, `r` (correlation coeff), `p_value`
- **Rows:** One per feature column
- **Description:** Correlation between ISC and each feature

#### `isc_vs_{feature}.png`
- **Type:** Scatter plot with regression line
- **Axes:** Feature value (x), ISC (y)
- **Description:** Visual representation of ISC-feature relationship

---

## Best Practices

1. **TR Consistency**: Ensure `tr_sec` matches the TR of preprocessed BOLD data. Mismatches will lead to incorrect TR index conversion.

2. **Annotation Quality**: Invest time in careful, consistent annotation. Quality of scene labels directly affects downstream statistical power.

3. **Feature Scaling**: If using custom features, consider normalizing to a standard scale (e.g., [0, 1]) for interpretability.

4. **Scene Duration**: Avoid very brief scenes (< 5 TRs). Use `min_scene_trs` to filter short segments.

5. **Feature Orthogonality**: If features are correlated, consider reporting correlations alongside ISC correlations.

6. **Sample Size**: ISC correlation analysis benefits from many subjects (n ≥ 8) and many scenes (n ≥ 20) for stability.

---

## Troubleshooting

### No Annotations Found

**Symptom:** Log message "No annotations found at {annotation_dir}. Creating templates."

**Solution:**
1. Check that `annotation_dir` exists and contains `*.csv` files
2. Ensure CSV filenames match `*.csv` pattern
3. Verify required columns (`onset_sec`, `offset_sec`) are present
4. Fill in the template CSV and place in `annotation_dir`

### n_volumes Warning

**Symptom:** "Could not determine n_volumes for {run_name} from runs_df..."

**Solution:**
1. Ensure `runs_df` (from `run_ingest()`) has an `n_volumes` column
2. Verify column is populated with integer values, not `NaN` or `None`
3. If using single-step runner, ensure metadata is passed correctly

### TR Index Out of Bounds

**Symptom:** Scene times exceed actual stimulus duration

**Solution:**
1. Verify `tr_sec` matches data TR exactly
2. Check annotation times don't exceed stimulus length
3. Adjust `min_scene_trs` if needed

---

## References

- **Scene Annotation Framework**: `src/fmri_pipeline/scene_annotation.py`
- **Pipeline Integration**: `src/fmri_pipeline/pipeline.py::run_scene_alignment_step()`
- **Configuration Loader**: `src/fmri_pipeline/config.py`
- **Example Config**: `config/pipeline.cneuromod_isc.yaml`

---

**Last Updated:** April 11, 2026
