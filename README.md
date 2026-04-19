# Reproducible fMRI Pipeline for Functional Brain Network Analysis

A modular, YAML-configured Python pipeline for multi-scale functional brain network analysis. Built for naturalistic (movie-viewing) and resting-state fMRI data, validated on independent datasets without code modification.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sarapradhan/biomed-research)

```
                    ┌─────────────────────────────────────────────────┐
                    │           YAML Configuration Layer              │
                    │  (pipeline.yaml / pipeline.ds007318.yaml / ...) │
                    └──────────────────────┬──────────────────────────┘
                                           │
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
    ┌─────────▼──────────┐   ┌─────────────▼─────────────┐   ┌────────▼─────────┐
    │  RELIABILITY LAYER │   │     FEATURE MODULES       │   │  DECISION LAYER  │
    │                    │   │                            │   │                  │
    │  Data Ingestion    │   │  ROI Time Series (Schaefer)│   │  Group Stats     │
    │  Preprocessing     │──▶│  Regional Homogeneity     │──▶│  (SZ vs HC)      │
    │  Quality Control   │   │  Static FC (Fisher-z)     │   │  Mass-Univariate │
    │  Motion Scrubbing  │   │  Dynamic FC (sliding-win) │   │  OLS + FDR       │
    │  Confound Regress. │   │  Spatial ICA              │   │                  │
    │  Band-pass Filter  │   │  PCA                      │   │  Sensitivity     │
    │                    │   │  ISC (leave-one-out)       │   │  Analysis        │
    └────────────────────┘   └───────────────────────────┘   └──────────────────┘
```

## Key Features

- **7 analysis modules** covering local synchrony, static/dynamic connectivity, decomposition, and intersubject correlation
- **YAML-driven configuration** — switch datasets, parameters, and analysis options without touching code
- **Cross-dataset portability** — validated on CNeuroMod Friends (naturalistic movie) and OpenNeuro ds007318 (task) without code changes
- **Sensitivity analysis** — built-in robustness benchmark testing GSR, parcellation, dFC windows, scrubbing thresholds, and smoothing kernels
- **Reproducibility** — deterministic seeds, timestamped logs, reusable manifests, fMRIPrep-compatible preprocessing

## Quickstart

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate fmri-unified-pipeline
pip install -r requirements.txt
```

### 2. Configure Paths

Copy the template and edit paths for your system:

```bash
cp config/pipeline.local.template.yaml config/pipeline.yaml
# Edit paths in config/pipeline.yaml to point to your data
```

### 3. Run the Pipeline

**Full pipeline (one command):**
```bash
python scripts/run_pipeline.py --config config/pipeline.yaml
```

**Step-by-step (milestone runs):**
```bash
python scripts/run_step.py --config config/pipeline.yaml --step preprocess_qc
python scripts/run_step.py --config config/pipeline.yaml --step roi_timeseries
python scripts/run_step.py --config config/pipeline.yaml --step group_stats
```

**Standalone ds007318 pilot (no BIDS index needed):**
```bash
python scripts/run_fmriprep_pilot.py \
    --data-root /path/to/fMRIPrep \
    --output-root /path/to/output
```

**Sensitivity analysis (robustness benchmark):**
```bash
python scripts/run_sensitivity_analysis.py \
    --data-root /path/to/fMRIPrep \
    --output-root /path/to/output
```

## Project Structure

```
biomed-research/
├── config/
│   ├── pipeline.yaml                 # Main config (SchizConnect)
│   ├── pipeline.ds007318.yaml        # OpenNeuro ds007318 pilot config
│   ├── pipeline.algonauts.yaml       # CNeuroMod naturalistic ISC config
│   ├── pipeline.cneuromod_isc.yaml   # Track 2: ISC extension config
│   ├── sensitivity.yaml              # Sensitivity analysis parameters
│   └── pipeline.local.template.yaml  # Template for local path setup
├── scripts/
│   ├── run_pipeline.py               # Full pipeline entry point
│   ├── run_step.py                   # Run individual pipeline steps
│   ├── run_fmriprep_pilot.py         # Standalone ds007318 runner
│   └── run_sensitivity_analysis.py   # Robustness benchmark runner
├── src/fmri_pipeline/
│   ├── pipeline.py                   # Pipeline orchestration
│   ├── config.py                     # YAML config loader
│   ├── bids_ingest.py                # BIDS data discovery
│   ├── preprocessing.py              # Confound regression, scrubbing, filtering
│   ├── qc.py                         # Quality control metrics and plots
│   ├── roi.py                        # Schaefer atlas ROI extraction
│   ├── reho.py                       # Regional homogeneity
│   ├── connectivity.py               # Static and dynamic FC
│   ├── ica.py                        # Spatial ICA + cross-subject matching
│   ├── pca_metrics.py                # PCA explained variance
│   ├── isc.py                        # Intersubject correlation
│   ├── stats.py                      # Group-level statistics (OLS, FDR)
│   ├── scene_annotation.py           # Scene annotation framework (ISC extension)
│   ├── viz.py                        # Visualization utilities
│   └── utils.py                      # Logging, seeds, path helpers
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── SOFTWARE_ARCHITECTURE.md
│   └── IMPLEMENTATION_RUNBOOK.md
├── LIMITATIONS.md                    # Known constraints and scope
├── ROADMAP.md                        # Future development plan
├── requirements.txt
└── environment.yml
```

## Method Summary

| Component | Implementation |
|-----------|---------------|
| Confounds | Friston-24 + WM/CSF + optional GSR |
| Scrubbing | FD > 0.5 mm (configurable) |
| Exclusion | >20% censored OR max translation >3mm OR max rotation >3 deg |
| Band-pass | 0.01–0.10 Hz with per-run TR from BIDS metadata |
| Smoothing | 6 mm FWHM Gaussian (configurable) |
| Atlas | Schaefer-200 (configurable: 100, 200, 400) |
| Static FC | Pearson correlation + Fisher z-transform |
| Dynamic FC | 30-TR sliding window, 5-TR step (configurable) |
| ISC | Leave-one-out with circular time-shift permutation null |
| Group stats | Mass-univariate OLS with FDR q<0.05 correction |

## Data Requirements

The pipeline expects fMRIPrep derivatives organized in BIDS format:

```
<derivatives_root>/<dataset>/fmriprep/
    sub-XX/ses-YY/func/
        sub-XX_ses-YY_task-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        sub-XX_ses-YY_task-*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
        sub-XX_ses-YY_task-*_desc-confounds_timeseries.tsv
```

## Validated Datasets

| Dataset | Type | Subjects | Use Case |
|---------|------|----------|----------|
| CNeuroMod Friends | Naturalistic movie-viewing | 4+ | ISC, dynamic FC, scene-linked analysis |
| OpenNeuro ds007318 | Working-memory task | 3 (5 runs) | Cross-dataset validation, sensitivity analysis |
| SchizConnect | Clinical resting-state | Variable | SZ vs HC group comparison |

## Documentation

- **[DeepWiki](https://deepwiki.com/sarapradhan/biomed-research)** — Interactive documentation: architecture, pipeline modules, datasets, and API reference
- **[Limitations](LIMITATIONS.md)** — Known constraints, scope boundaries, and methodological caveats
- **[Roadmap](ROADMAP.md)** — Planned extensions and future development
- **[Project Overview](docs/PROJECT_OVERVIEW.md)** — Scientific motivation and design rationale
- **[Software Architecture](docs/SOFTWARE_ARCHITECTURE.md)** — Module design and data flow
- **[Implementation Runbook](docs/IMPLEMENTATION_RUNBOOK.md)** — Step-by-step setup guide

## Citation

If you use this pipeline in your work, please cite:

> Pradhan, S. (2026). Building a Reproducible Pipeline for Functional Brain Network Analysis During Naturalistic Movie Viewing

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
