# Implementation and Operations Runbook

## 1) Repository Map
- `config/`: runtime configuration (`pipeline.yaml`)
- `src/fmri_pipeline/`: core implementation modules
- `scripts/`: CLI entrypoints
- `tests/`: lightweight sanity tests
- `docs/`: project documentation

## 2) Milestone-to-Module Mapping
1. Ingest
   - `bids_ingest.py`
2. Preprocess + QC
   - `preprocessing.py`, `qc.py`
3. ROI
   - `roi.py`
4. ReHo
   - `reho.py`
5. Static FC / Dynamic FC
   - `connectivity.py`
6. ICA
   - `ica.py` + orchestration in `pipeline.py`
7. PCA
   - `pca_metrics.py`
8. ISC
   - `isc.py`
9. Group stats and visualizations
   - `stats.py`, `viz.py`, `pipeline.py`

## 3) Standard Execution
### Full pipeline
```bash
python scripts/run_pipeline.py --config config/pipeline.yaml
```

### Stepwise execution
```bash
python scripts/run_step.py --config config/pipeline.yaml --step ingest
python scripts/run_step.py --config config/pipeline.yaml --step preprocess_qc
python scripts/run_step.py --config config/pipeline.yaml --step roi_timeseries
python scripts/run_step.py --config config/pipeline.yaml --step reho
python scripts/run_step.py --config config/pipeline.yaml --step static_dynamic_fc
python scripts/run_step.py --config config/pipeline.yaml --step ica
python scripts/run_step.py --config config/pipeline.yaml --step pca
python scripts/run_step.py --config config/pipeline.yaml --step isc
python scripts/run_step.py --config config/pipeline.yaml --step group_stats
```

## 4) Mandatory Configuration Checks Before Running
- `paths.bids_roots`: must exist and be readable
- `paths.derivatives_root`: must contain fMRIPrep derivatives under expected dataset subfolders
- `bids.task_rest` and `bids.task_movie`: must match actual BIDS task labels
- `stats.diagnosis_column`: must match participants/phenotypic column name
- `stats.patient_label` and `stats.control_label`: must match exact label values

## 5) Runtime Artifacts by Milestone
### Ingest
- `manifests/participants_merged.csv`
- `manifests/run_manifest_raw.csv`

### Preprocess/QC
- `preprocessed/sub-*/.../clean_unsmoothed_bold.nii.gz`
- `preprocessed/sub-*/.../clean_smoothed6mm_bold.nii.gz`
- `qc/qc_summary.csv`
- `manifests/run_manifest_preprocessed.csv`

### ROI/ReHo/FC/ICA/PCA/ISC
- ROI: `roi_timeseries.npy/csv`
- ReHo: `reho_map.nii.gz`
- Static FC: `static_fc_fisherz.npy/csv`
- Dynamic FC: `dfc_mean.npy/csv`, `dfc_variability_std.npy/csv`
- ICA: `ica_subject_loadings.csv`, `ica_component_matching.csv`
- PCA: `pca_explained_variance.csv`
- ISC: `isc_mean.nii.gz`, `isc_qvals.nii.gz`, `isc_sig_fdrq05.nii.gz` (if movie cohort available)

### Group stats
- Edge tables in `tables/`
- Voxel maps in `group_stats/`
- Figures in `figures/`

## 6) Statistical Conventions
- Diagnosis contrast coding: patient vs control from config labels
- Positive diagnosis beta means higher value in patient group
- Multiple testing:
  - edge-wise: BH-FDR (q<0.05)
  - voxelwise: FDR maps currently generated in code

## 7) QC Acceptance Guidance
Minimum checks before trusting group-level output:
- `qc_summary.csv`: confirm exclusion flags and motion distributions
- verify no group has systematically higher mean FD without proper covariate control
- confirm subject counts in manifests are consistent through each milestone

## 8) Performance and Scaling
- ReHo is the most compute-heavy voxelwise step
- ICA convergence can vary by data quality and number of components
- Use debug mode for fast validation:
  - `project.debug_mode: true`
  - `project.debug_subject_limit: <small integer>`

## 9) Common Failure Modes and Fixes
- No runs found:
  - wrong task label, wrong derivatives path, or missing BIDS entities
- Merge type mismatch in group stats:
  - subject IDs inconsistent (e.g., `01` vs `1`) between files
- ISC skipped:
  - insufficient movie subjects after filtering/exclusion
- ICA convergence warnings:
  - increase `ica.max_iter`, consider lower component count in debug

## 10) Logging and Audit Trail
- Main logs: `derivatives/logs/pipeline.log`
- Per-step logs: created by step runner names
- Persisted manifests serve as stage-level audit checkpoints

## 11) Recommended Next Documentation Additions
- Data dictionary for all generated tables
- Reproducibility report template (software versions + hash of config)
- Figure interpretation guide for collaborators
