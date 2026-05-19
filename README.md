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
    ┌─────────▼──────────┐   ┌─────────────▼─────────────┐   ┌────────▼──────────────┐
    │  RELIABILITY LAYER │   │     FEATURE MODULES       │   │  DECISION LAYER       │
    │                    │   │                            │   │                       │
    │  Data Ingestion    │   │  ROI Time Series (Schaefer)│   │  Group Stats          │
    │  Preprocessing     │──▶│  Regional Homogeneity     │──▶│  (SZ vs HC)           │
    │  Quality Control   │   │  Static FC (Fisher-z)     │   │  Mass-Univariate      │
    │  Motion Scrubbing  │   │  Dynamic FC (sliding-win) │   │  OLS + FDR            │
    │  Confound Regress. │   │  Spatial ICA              │   │                       │
    │  Band-pass Filter  │   │  PCA                      │   │  Reproducibility      │
    │                    │   │  ISC (leave-one-out)       │   │  Analyses             │
    └────────────────────┘   └───────────────────────────┘   └───────────────────────┘
```

## Key Features

- **8 analysis modules** covering local synchrony, static/dynamic connectivity, decomposition, intersubject correlation, and a full reproducibility validation suite
- **YAML-driven configuration** — switch datasets, parameters, and analysis options without touching code
- **Cross-dataset portability** — validated on CNeuroMod Friends (naturalistic movie) and OpenNeuro ds007318 (task/resting-state) without code changes
- **Reproducibility suite** — six dedicated validation analyses (FC within/between, ReHo stability, ICA stability, graph metric bootstrap, dynamic FC window sensitivity, canonical network anchor)
- **Sensitivity analysis** — built-in robustness benchmark testing GSR, parcellation, dFC windows, scrubbing thresholds, and smoothing kernels
- **Deterministic by default** — pinned seeds, timestamped manifests, fMRIPrep-compatible preprocessing

## Quickstart

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate fmri-unified-pipeline
pip install -e .
```

`pyproject.toml` is the single source of truth for dependencies. `requirements.txt` is an auto-generated, fully pinned lockfile produced by `pip-compile` (from [pip-tools](https://pip-tools.readthedocs.io)) — useful for reproducing an exact environment with `pip install -r requirements.txt`. Do not edit `requirements.txt` by hand; to refresh it after changing `pyproject.toml`, run:

```bash
pip install pip-tools
pip-compile --output-file=requirements.txt pyproject.toml
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

**Reproducibility validation suite (JEI Table 1):**
```bash
# One-command run (recommended):
bash scripts/run_all_reproducibility.sh

# Or step by step:
python scripts/prep_reproducibility_inputs.py
python scripts/run_reproducibility.py --config config/reproducibility_real.yaml
cat reports/reproducibility/scorecard.md
```

## Project Structure

```
biomed-research/
├── config/
│   ├── pipeline.yaml                 # Main config (SchizConnect)
│   ├── pipeline.ds007318.yaml        # OpenNeuro ds007318 pilot config
│   ├── pipeline.algonauts.yaml       # CNeuroMod naturalistic ISC config
│   ├── pipeline.cneuromod_isc.yaml   # Track 2: ISC extension config
│   ├── pipeline.local.template.yaml  # Template for local path setup
│   ├── sensitivity.yaml              # Sensitivity analysis parameters
│   ├── reproducibility.yaml          # Reproducibility suite (synthetic)
│   └── reproducibility_real.yaml     # Reproducibility suite (ds007318)
├── scripts/
│   ├── run_pipeline.py               # Full pipeline entry point
│   ├── run_step.py                   # Run individual pipeline steps
│   ├── run_fmriprep_pilot.py         # Standalone ds007318 runner
│   ├── run_sensitivity_analysis.py   # Robustness benchmark runner
│   ├── run_sensitivity_on_mac.sh     # Mac-local sensitivity helper
│   ├── prep_reproducibility_inputs.py# Stage data for reproducibility suite
│   ├── run_reproducibility.py        # Reproducibility suite runner
│   ├── run_all_reproducibility.sh    # One-command reproducibility run
│   ├── build_scorecard.py            # Generate Table 1 scorecard
│   ├── data_inventory.py             # Audit input data availability
│   ├── organize_schizconnect.py      # SchizConnect file organizer
│   ├── run_dfc_sensitivity.py        # Dynamic FC window sweep
│   ├── run_ica_stability.py          # ICA seed stability runner
│   ├── run_isc_extension.py          # ISC extension runner
│   ├── run_network_anchor.py         # Canonical network anchor runner
│   └── run_reho_stability.py         # ReHo run-to-run stability runner
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
│   ├── isc.py                        # Intersubject correlation (ISC)
│   ├── stats.py                      # Group-level statistics (OLS, FDR)
│   ├── scene_annotation.py           # Scene annotation framework (ISC extension)
│   ├── viz.py                        # Visualization utilities
│   ├── utils.py                      # Logging, seeds, path helpers
│   └── reproducibility/              # Reproducibility validation suite
│       ├── fc_reproducibility.py     # FC within vs between-subject similarity
│       ├── reho_stability.py         # ReHo run-to-run stability
│       ├── ica_stability.py          # ICA seed and LORO-CV stability
│       ├── graph_stability.py        # Graph metric bootstrap + LORO-CV
│       ├── dfc_sensitivity.py        # Dynamic FC window-size sensitivity
│       ├── network_anchor.py         # Canonical 7-network biological anchor
│       └── scorecard.py              # Table 1 pass/fail scorecard
├── tests/
│   ├── test_preprocessing.py
│   ├── test_connectivity.py
│   ├── test_isc.py
│   ├── test_stats.py
│   ├── test_pca_and_roi.py
│   ├── test_viz_and_reho.py
│   ├── test_scene_annotation.py
│   ├── test_config_and_utils.py
│   ├── test_sensitivity.py
│   ├── test_sanity.py
│   ├── test_fc_reproducibility.py
│   ├── test_ica_stability.py
│   ├── test_graph_stability.py
│   ├── test_dfc_sensitivity.py
│   ├── test_network_anchor.py
│   ├── test_reho_stability.py
│   ├── test_scorecard.py
│   └── test_run_reproducibility.py
├── reports/
│   └── reproducibility/              # Reproducibility suite outputs
│       ├── scorecard.md / .csv       # Table 1 summary
│       ├── fc_within_vs_between.csv
│       ├── reho_summary.csv
│       ├── ica_stability_seeds.csv
│       ├── ica_stability_lorocv.csv
│       ├── graph_metrics_bootstrap.csv
│       ├── dfc_sensitivity.json
│       └── network_anchor_summary.csv
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── SOFTWARE_ARCHITECTURE.md
│   └── IMPLEMENTATION_RUNBOOK.md
├── REPRODUCIBILITY_RUNBOOK.md        # Step-by-step reproducibility guide
├── LIMITATIONS.md                    # Known constraints and scope
├── CONTRIBUTING.md                   # Contribution guidelines
├── requirements.txt
└── environment.yml
```

## Method Summary

| Component | Implementation |
|-----------|---------------|
| Confounds | Friston-24 + WM/CSF + optional GSR |
| Scrubbing | FD > 0.5 mm (configurable) |
| Exclusion | >20% censored OR max translation >3 mm OR max rotation >3 deg |
| Band-pass | 0.01–0.10 Hz with per-run TR from BIDS metadata |
| Smoothing | 6 mm FWHM Gaussian (configurable) |
| Atlas | Schaefer-200 (configurable: 100, 200, 400) |
| Static FC | Pearson correlation + Fisher z-transform |
| Dynamic FC | 30-TR sliding window, 5-TR step (configurable) |
| ISC | Leave-one-out with circular time-shift permutation null |
| Group stats | Mass-univariate OLS with FDR q<0.05 correction |

## Reproducibility Suite (Table 1)

Six analyses validate pipeline stability on OpenNeuro ds007318 (N=3, 5 runs total):

| Analysis | Module | Pass Criterion |
|----------|--------|---------------|
| FC within > between-subject | `fc_reproducibility` | Bootstrap CI on gap excludes zero |
| ReHo run-to-run stability | `reho_stability` | Mean r > 0.70 across run pairs |
| ICA component recovery | `ica_stability` | ≥18/20 components robust across seeds |
| Graph metric bootstrap | `graph_stability` | Modularity CV < 20% |
| Dynamic FC window sensitivity | `dfc_sensitivity` | ARI > 0.30 across window sizes |
| Canonical network anchor | `network_anchor` | Within > between-network FC, p < 0.05 |

See [REPRODUCIBILITY_RUNBOOK.md](REPRODUCIBILITY_RUNBOOK.md) for full instructions.

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

| Dataset | Type | Subjects | Runs | Use Case |
|---------|------|----------|------|----------|
| CNeuroMod Friends | Naturalistic movie-viewing | 4+ | Multiple | ISC, dynamic FC, scene-linked analysis |
| OpenNeuro ds007318 | Working-memory (pseudo-resting) | 3 | 5 total | Reproducibility validation, sensitivity analysis |
| SchizConnect | Clinical resting-state | Variable | Variable | SZ vs HC group comparison (code-ready; awaiting data) |

> **Note on ds007318:** Participants are drawn from a clinical population (working-memory removal paradigm, Northwest Normal University). The dataset contains no healthy control arm; group-level statistics are disabled for this dataset. Results are treated as feasibility demonstrations.

## Documentation

- **[DeepWiki](https://deepwiki.com/sarapradhan/biomed-research)** — Interactive documentation: architecture, pipeline modules, datasets, and API reference
- **[Reproducibility Runbook](REPRODUCIBILITY_RUNBOOK.md)** — Step-by-step guide to running the validation suite
- **[Limitations](LIMITATIONS.md)** — Known constraints, scope boundaries, and methodological caveats
- **[Project Overview](docs/PROJECT_OVERVIEW.md)** — Scientific motivation and design rationale
- **[Software Architecture](docs/SOFTWARE_ARCHITECTURE.md)** — Module design and data flow
- **[Implementation Runbook](docs/IMPLEMENTATION_RUNBOOK.md)** — Step-by-step setup and execution guide
- **[Contributing](CONTRIBUTING.md)** — How to add modules, report bugs, and submit PRs

## Citation

If you use this pipeline in your work, please cite:

> Pradhan, S. (2026). Building a Reproducible Pipeline for Functional Brain Network Analysis During Naturalistic Movie Viewing. *Journal of Emerging Investigators.* https://github.com/sarapradhan/biomed-research

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
