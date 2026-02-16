# Project Overview

## What This Project Does
This repository implements an end-to-end, BIDS-native fMRI analysis platform for two study settings:
- Resting-state schizophrenia vs healthy controls (SchizConnect-style cohorts)
- Naturalistic movie fMRI ISC in healthy controls (Algonauts-style data)

The pipeline intentionally stops at group-level results. It does not perform symptom prediction, clinical outcome modeling, or any downstream machine-learning classifier beyond the requested inferential metrics.

## Why It Is Structured This Way
The system follows a modular product-platform pattern:
- Reliability layer: preprocessing + quality control so all downstream features consume standardized, motion-screened signals.
- Feature modules: independent metric generators (ROI, ReHo, static FC, dFC, ICA, PCA, ISC).
- Decision layer: group-level inference and correction that turns features into interpretable findings.

This design allows adding/removing metrics without rewriting ingestion or group statistics.

## Milestones Implemented
1. BIDS ingestion and harmonization
2. Preprocessing and QC
3. ROI time-series extraction (Schaefer-200)
4. ReHo map computation
5. Static FC (Pearson + Fisher-z)
6. Dynamic FC (30 TR / 5 TR)
7. Subject ICA + cross-subject matching
8. Subject PCA (top-5 explained variance)
9. ISC leave-one-out + permutation null
10. Group-level statistics + visualization/tables

## Core Scientific Rules Enforced
- Confounds: Friston-24 + WM/CSF (if present)
- Scrubbing threshold: FD > 0.5 mm
- Exclusion: >20% censored OR max translation >3 mm OR max rotation >3 deg
- Temporal filtering: 0.01-0.10 Hz (TR-aware)
- Spatial smoothing: 6 mm FWHM
- Atlas: Schaefer-200
- Edge-level multiple testing: FDR q<0.05
- Voxelwise correction: FDR-based maps in current implementation

## Inputs and Outputs
### Inputs
- BIDS roots (raw)
- fMRIPrep derivatives (required at runtime)
- Optional phenotypic TSV (diagnosis + covariates)

### Outputs
Under `derivatives/metrics`:
- `manifests/`: reusable run/subject manifests between milestones
- `qc/`: run-level QC tables and plots
- metric folders: `roi_timeseries`, `reho`, `static_fc`, `dynamic_fc`, `ica`, `pca`, `isc`
- `tables/`: group-level inferential tables
- `figures/`: group-level figures
- `group_stats/`: voxelwise statistical maps

## Current Local Run Context
Recent execution on your machine used data mapped under:
- `/Users/rahul.pradhan/AppDev/Schizo/fMRI`

In that run:
- Resting-state milestones completed through group-level outputs
- ISC was skipped due to insufficient movie subjects in the available dataset
