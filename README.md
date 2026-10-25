# Reproducible fMRI Pipeline for Functional Brain Network Analysis

A modular, YAML-configured Python pipeline for multi-scale functional brain network analysis. Built for naturalistic (movie-viewing) and resting-state fMRI data. Developed on a CNeuroMod *Friends* subset and validated, without code modification, on an independent public dataset (OpenNeuro ds007318).

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

- **7 feature modules** covering local synchrony (ReHo), static and dynamic connectivity, spatial ICA, PCA, and intersubject correlation (matching the diagram above: `roi`, `reho`, `connectivity`, `ica`, `pca_metrics`, `isc`, plus `scene_annotation` for optional narrative-aligned ISC analysis), backed by a dedicated reproducibility validation suite — see [Software Architecture](docs/SOFTWARE_ARCHITECTURE.md) for the full per-file breakdown (15 core modules including ingestion/QC/stats + a 7-file reproducibility subpackage)
- **YAML-driven configuration** — switch datasets, parameters, and analysis options without touching code
- **Cross-dataset portability** — developed on CNeuroMod *Friends* (naturalistic movie viewing), then validated on OpenNeuro ds007318 (working-memory task, analyzed as pseudo-resting-state) without code changes
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

**Reproducibility validation suite (manuscript Table 1):**
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
│   ├── pipeline.yaml                 # Main config (default parameters; set paths for your data)
│   ├── pipeline.example.yaml         # Annotated example config
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
│   ├── run_reho_stability.py         # ReHo run-to-run stability runner
│   ├── fc_fingerprint.py             # Edge-level network-anchor corroboration (manuscript)
│   ├── fc_matrix_figure.py           # Group-average FC matrix figure (manuscript)
│   ├── ica_label_components.py       # ICA component network labeling (manuscript)
│   └── reho_regional_profile.py      # ReHo regional profile summary (manuscript)
├── src/fmri_pipeline/
│   ├── pipeline.py                   # Pipeline orchestration
│   ├── config.py                     # YAML config loader
│   ├── bids_ingest.py                # BIDS data discovery
│   ├── preprocessing.py              # Confound regression, scrubbing, filtering
│   ├── qc.py                         # Quality control metrics and plots
│   ├── roi.py                        # Schaefer atlas ROI extraction
│   ├── reho.py                       # Regional homogeneity
│   ├── connectivity.py               # Static and dynamic FC
│   ├── ica.py                        # Spatial ICA + hierarchical cross-subject matching
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
├── reports/sensitivity/              # Sensitivity analysis outputs (11 conditions)
├── figures/                          # Manuscript figures
├── sensitivity_figures_real/         # Combined sensitivity figure
├── paper.md / paper.tex / paper.bib  # Manuscript source (unpublished)
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

Six analyses validate pipeline stability on OpenNeuro ds007318 (N=3, 5 runs total). The FC-reproducibility and ReHo pass criteria below are met by bootstrap CIs built from only 2 within-subject run pairs — the manuscript reports these as **suggestive, not confirmatory** evidence, and the current status labels below should be read with that caveat, not as precise interval estimates:

| Analysis | Module | Pass Criterion | Status |
|----------|--------|---------------|:---:|
| FC within > between-subject | `fc_reproducibility` | Bootstrap CI on gap excludes zero | ✓ (suggestive, n=2) |
| ReHo run-to-run stability | `reho_stability` | Mean r > 0.70 across run pairs | ✓ (suggestive, n=2) |
| ICA seed stability | `ica_stability` | ≥18/20 components robust across seeds | ✓ (seed only) |
| ICA cross-run stability (LORO-CV) | `ica_stability` | ≥18/20 components robust across 3 leave-one-run-out subsets | ✗ fails (mean \|r\|=0.41, N≤3 subsets is underpowered) |
| Graph metric bootstrap | `graph_stability` | Modularity CV < 20% | ✓ |
| Dynamic FC window sensitivity | `dfc_sensitivity` | ARI > 0.30 across window sizes | n/a (exploratory) |
| Canonical network anchor | `network_anchor` | Within > between-network FC, p < 0.05 | ✓ (spatial autocorrelation not ruled out — see Limitations) |

Seed stability and cross-run (LORO-CV) stability of the ICA temporal-ICA proxy are tracked as separate rows so a strong seed result cannot mask the weaker cross-run result. The generated `scorecard.md` labels the cross-run row "n/a (underpowered)"; this README reports it as a failure so the result is not read as passing. See [REPRODUCIBILITY_RUNBOOK.md](REPRODUCIBILITY_RUNBOOK.md) for full instructions.

## Data Requirements

The pipeline expects fMRIPrep derivatives organized in BIDS format:

```
<derivatives_root>/<dataset>/fmriprep/
    sub-XX/ses-YY/func/
        sub-XX_ses-YY_task-*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
        sub-XX_ses-YY_task-*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
        sub-XX_ses-YY_task-*_desc-confounds_timeseries.tsv
```

## Datasets

Only OpenNeuro ds007318 has published validation results in this repository.

| Dataset | Type | Subjects | Runs | Role |
|---------|------|----------|------|------|
| CNeuroMod Friends | Naturalistic movie-viewing | Subset | Subset | Pipeline development; not separately validated |
| OpenNeuro ds007318 | Working-memory (pseudo-resting) | 3 | 5 total | Validation: reproducibility suite and sensitivity analysis |
| SchizConnect | Clinical resting-state | Variable | Variable | SZ vs HC group comparison (code-ready; awaiting data) |

> **Note on ds007318:** Participants are drawn from a clinical population (working-memory removal paradigm, Northwest Normal University). The dataset contains no healthy control arm; group-level statistics are disabled for this dataset. Results are treated as feasibility demonstrations.

## Documentation

- **[DeepWiki](https://deepwiki.com/sarapradhan/biomed-research)** — Interactive documentation generated automatically by a third-party AI tool; it may lag behind or differ from this repository
- **[Reproducibility Runbook](REPRODUCIBILITY_RUNBOOK.md)** — Step-by-step guide to running the validation suite
- **[Limitations](LIMITATIONS.md)** — Known constraints, scope boundaries, and methodological caveats
- **[Project Overview](docs/PROJECT_OVERVIEW.md)** — Scientific motivation and design rationale
- **[Software Architecture](docs/SOFTWARE_ARCHITECTURE.md)** — Module design and data flow
- **[Implementation Runbook](docs/IMPLEMENTATION_RUNBOOK.md)** — Step-by-step setup and execution guide
- **[Contributing](CONTRIBUTING.md)** — How to add modules, report bugs, and submit PRs

## Citation

If you use this pipeline in your work, please cite the archived software:

> Pradhan, S. (2026). *fmri-pipeline* (Version 1.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20280071

The accompanying manuscript, *A Configuration-Driven Python Pipeline for Reproducible Functional Brain Network Analysis: Internal Validation on a Public fMRI Dataset*, is unpublished. Its source is in `paper.md`.

## Acknowledgements

The author thanks Arefeh Sherafati, Research Scientist at St. Jude Children's Research Hospital, for guidance and feedback, and the team that collected and shared OpenNeuro ds007318.

Generative AI (Anthropic's Claude) was used as a writing and coding assistant. All scientific content, analyses, interpretations, and final code were designed, reviewed, and verified by the author.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
