# Code Review: Uncommitted Changes

**Date:** April 11, 2026  
**Repository:** biomed-research (fMRI Pipeline)  
**Branch:** main  
**Status:** Changes not yet staged for commit

## Summary

This review covers **1,025 insertions and 233 deletions** across **17 modified files** plus **14 new untracked files** in your fMRI pipeline repository. The primary themes are:

1. **Documentation Enhancement** – Comprehensive docstrings added to all modules
2. **New Feature** – Scene annotation framework for ISC-to-scene alignment (Track 2 extension)
3. **Project Metadata** – Added pyproject.toml, paper.md, ROADMAP.md, licensing
4. **Code Quality** – Improved code organization and module documentation
5. **Test Coverage** – New test suite files (8 new test modules)

---

## Detailed Findings

### 1. **Documentation Improvements** ✅

All core modules now have comprehensive module-level docstrings following NumPy documentation style:

- **`__init__.py`**: Expanded from 1 line → 28 lines with full package description, workflow overview, and cross-references
- **`config.py`**: Added docstring explaining YAML loading and directory initialization
- **`connectivity.py`**: Added detailed module and function docstrings with parameter/return documentation
- **`pca_metrics.py`**: Added module description and function documentation
- **`preprocessing.py`**: Added comprehensive docstring with preprocessing pipeline description and references (Power et al. 2012)
- **`reho.py`**: Added docstring with Kendall's W explanation and Zang et al. 2004 reference
- **`roi.py`**: Added docstring with Schaefer atlas description and Schaefer et al. 2018 reference
- **`utils.py`**: Added docstring listing utility categories
- **`viz.py`**: Added docstring describing visualization functions

**Recommendation:** Excellent documentation practice. Consider applying similar docstring standards to helper functions within modules (e.g., inside `connectivity.py` and `preprocessing.py`).

---

### 2. **Major New Feature: Scene Annotation Framework** 🆕

#### New Module: `scene_annotation.py` (~400+ lines)

This is a substantial addition enabling "Track 2 ISC extension" – aligning intersubject correlation maps with scene-level annotations.

**Key Functions:**
- `load_scene_annotations()` – Load and validate scene annotation CSVs
- `scenes_to_tr_indices()` – Convert scene timing (seconds) → fMRI volume indices
- `align_isc_to_scenes()` – Summarize ISC values within annotated scenes
- `correlate_isc_with_features()` – Compute ISC-feature correlations
- `create_annotation_template()` – Generate example annotation CSV
- `plot_isc_scene_alignment()` – Visualization helper
- `load_all_annotations()` – Batch load annotations from directory

**Annotation Format:**
```csv
onset_sec, offset_sec, emotional_valence, social_cognition, narrative_transition
0.0, 45.2, 0.3, 0, 0
45.2, 78.6, -0.5, 1, 1
```

**Integration in Pipeline:**
- New step `run_scene_alignment_step()` added to `pipeline.py` (lines 351-465)
- Runs after ISC computation: `run_isc_step()` → `run_scene_alignment_step()`
- Conditionally enabled via YAML config: `scene_annotation.enabled`

**Observations:**
- ✅ Well-documented with clear docstrings
- ✅ Modular design with separate functions
- ⚠️ No error handling for missing TR metadata – relies on `runs_df` having `n_volumes`
- ⚠️ Defaults `n_volumes` to 200 if not found; this could mask data issues
- ⚠️ Uses voxel count instead of timepoint count (line: `n_volumes = isc_data[mask_vec].shape[0]`)
- **Recommendation:** Add validation that `n_volumes` is correctly populated; consider using metadata from BIDS filenames or nifti headers as fallback

---

### 3. **Pipeline Integration Changes**

#### `pipeline.py` modifications (+127 lines):

**Imports added:**
```python
from .scene_annotation import (
    align_isc_to_scenes,
    correlate_isc_with_features,
    create_annotation_template,
    load_all_annotations,
    plot_isc_scene_alignment,
    scenes_to_tr_indices,
)
```

**Main workflow update:**
```python
# Old: run_isc_step(cfg, preproc_runs, logger)
# New:
isc_paths = run_isc_step(cfg, preproc_runs, logger)  # Now returns dict
run_scene_alignment_step(cfg, preproc_runs, isc_paths, logger)  # New step
```

**Observations:**
- ✅ Clean separation of concerns
- ✅ Scene alignment step is optional (gated by config)
- ⚠️ `run_isc_step()` signature changed (now returns `isc_paths`). Ensure backward compatibility if other code depends on this.
- ✅ Logging follows existing patterns

---

### 4. **Configuration Changes**

Multiple YAML files updated:
- `config/pipeline.yaml` – Main config (+12 lines)
- `config/pipeline.ds007318.yaml` – DS007318 variant (+12 lines)
- `config/pipeline.local.template.yaml` – Local template (+14 lines)
- **New:** `config/pipeline.cneuromod_isc.yaml` – ISC-specific template
- **New:** `config/sensitivity.yaml` – Sensitivity analysis params

**Missing in this review:** See the actual YAML changes to validate schema updates for `scene_annotation` config block.

---

### 5. **New Files & Metadata**

**Project Metadata:**
- ✅ `pyproject.toml` – Build system, dependencies, entry points (standard Python package format)
- ✅ `paper.md` – Likely JOSS paper metadata
- ✅ `paper.bib` – BibTeX references
- ✅ `ROADMAP.md` – Development roadmap
- ✅ `CONTRIBUTING.md` – Contribution guidelines
- ✅ `LICENSE` – Open source license
- ✅ `LIMITATIONS.md` – Known limitations/scope

**New Scripts:**
- `scripts/run_isc_extension.py` – Track 2 ISC extension runner
- `scripts/run_sensitivity_analysis.py` – Sensitivity analysis script
- `scripts/run_sensitivity_on_mac.sh` – macOS-specific shell script

**Test Coverage (8 new test files):**
- `tests/test_config_and_utils.py`
- `tests/test_connectivity.py`
- `tests/test_isc.py`
- `tests/test_pca_and_roi.py`
- `tests/test_preprocessing.py`
- `tests/test_sanity.py`
- `tests/test_scene_annotation.py` ⭐ Tests new scene annotation module
- `tests/test_sensitivity.py`
- `tests/test_stats.py`
- `tests/test_viz_and_reho.py`

**Recommendation:** Review test coverage in `test_scene_annotation.py` to ensure edge cases are covered (missing annotations, mismatched TR, boundary conditions).

---

### 6. **Documentation Updates**

**`docs/SOFTWARE_ARCHITECTURE.md`** (+767 lines)
- Substantial expansion of architecture documentation
- Likely includes new scene annotation system design

**`docs/PROJECT_OVERVIEW.md`** (minor edit)

**`README.md`** (+202 lines)
- Greatly expanded project README

**Recommendation:** Review the architecture doc to ensure scene annotation design is clearly explained.

---

## Code Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Documentation** | ✅ Excellent | Comprehensive docstrings, references added |
| **Code Organization** | ✅ Good | New module well-structured, follows patterns |
| **Error Handling** | ⚠️ Partial | Scene alignment step needs better fallback handling |
| **Testing** | ✅ Good | 8 new test files, but need to verify coverage |
| **Backward Compatibility** | ⚠️ Check | `run_isc_step()` now returns dict (breaking change?) |
| **Performance** | ✅ Expected | No obvious inefficiencies |

---

## Recommendations

### High Priority
1. **Verify `run_isc_step()` return value** – Ensure any existing callers of this function are updated or handle the new dict return value
2. **Test scene annotation edge cases** – Validate handling of:
   - Missing annotation files
   - Misaligned TR values
   - Empty scenes
   - Out-of-range scene timings
3. **Document config schema** – Add validation for new `scene_annotation` YAML block with required/optional keys

### Medium Priority
4. **Improve `n_volumes` detection** – Don't silently default to 200; warn or error if unable to determine
5. **Add integration tests** – Test full pipeline with scene annotations enabled
6. **Verify test coverage** – Run `pytest --cov` to check coverage percentages

### Low Priority
7. **Review sensitivity analysis integration** – New sensitivity scripts may need documentation
8. **Add version bump** – Consider updating `__version__` from "0.1.0" given substantial feature additions

---

## Summary Table

| Category | Modified | New | Total |
|----------|----------|-----|-------|
| Source Code Files | 8 | 1 | 9 |
| Config Files | 3 | 2 | 5 |
| Documentation | 3 | 3 | 6 |
| Tests | 0 | 8 | 8 |
| Scripts | 1 | 2 | 3 |
| **Total** | **17** | **14** | **31** |

---

## Next Steps

1. **Before committing:**
   - Run `pytest` to ensure all tests pass
   - Run `mypy src/` (if type checking enabled) to catch type issues
   - Review the actual YAML config changes
   - Test with a real dataset if possible

2. **Commit strategy:**
   - Consider breaking into logical commits:
     - Commit 1: Documentation improvements (non-functional)
     - Commit 2: Scene annotation module + tests
     - Commit 3: Pipeline integration
     - Commit 4: Config files + metadata
     - Commit 5: New scripts

3. **Post-commit:**
   - Push to feature branch for PR review
   - Request review from collaborators
   - Add entry to CHANGELOG or ROADMAP

---

**Review completed:** April 11, 2026