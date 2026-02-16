# Unified fMRI Pipeline (SchizConnect + Algonauts 2025)

Reproducible Python 3.8 pipeline for:
- Resting-state schizophrenia vs healthy-control analyses (SchizConnect)
- Naturalistic movie ISC in controls (Algonauts 2025 Friends)
- Output scope stops at group-level statistics

## Platform Design Analogy
- Reliability layer: preprocessing + QC (motion censoring, confounds, filter, smoothing, exclusion rules)
- Feature modules: ROI, ReHo, static FC, dynamic FC, ICA, PCA, ISC
- Decision layer: group-level inference with covariates and multiple-comparison correction

## Documentation
- Project overview: `docs/PROJECT_OVERVIEW.md`
- Software architecture: `docs/SOFTWARE_ARCHITECTURE.md`
- Implementation runbook: `docs/IMPLEMENTATION_RUNBOOK.md`

## Quickstart

### 1) Environment
```bash
conda env create -f environment.yml
conda activate fmri-unified-pipeline
pip install -r requirements.txt
```

### 2) Configure paths
Copy and edit config:
```bash
cp config/pipeline.local.template.yaml config/pipeline.yaml
```

### 3) One-command run
```bash
python scripts/run_pipeline.py --config config/pipeline.yaml
```

### 4) Milestone run
```bash
python scripts/run_step.py --config config/pipeline.yaml --step preprocess_qc
python scripts/run_step.py --config config/pipeline.yaml --step roi_timeseries
python scripts/run_step.py --config config/pipeline.yaml --step group_stats
```

## Data Expectations
- BIDS roots:
  - `/Users/rahul.pradhan/AppDev/Schizo/fMRI/data/schizconnect`
  - `/Users/rahul.pradhan/AppDev/Schizo/fMRI/data/algonauts_friends`
- fMRIPrep derivatives under:
  - `/Users/rahul.pradhan/AppDev/Schizo/fMRI/derivatives/<dataset>/fmriprep`
- Optional phenotypic TSV with diagnosis and covariates:
  - `/Users/rahul.pradhan/AppDev/Schizo/fMRI/BDI/phenotypic.tsv`

If derivatives are absent, use the command template saved to:
- `derivatives/metrics/repro/fmriprep_command_template.json`

Example Docker command pattern:
```bash
docker run --rm -ti \
  -v /path/to/bids:/data:ro \
  -v /path/to/derivatives:/out \
  -v /path/to/work:/work \
  nipreps/fmriprep:latest /data /out participant \
  --use-aroma --output-spaces MNI152NLin2009cAsym \
  --fs-no-reconall --nthreads 16 --omp-nthreads 8 -w /work
```

## Method Constraints Implemented
- Confounds: Friston-24 + WM/CSF (if available)
- Scrubbing: FD > 0.5mm
- Exclusion: >20% censored OR max translation >3mm OR max rotation >3 degrees
- Band-pass: 0.01-0.10 Hz with TR from BIDS metadata per run
- Smoothing: 6mm FWHM (saved alongside unsmoothed cleaned images)
- Atlas: Schaefer-200
- dFC windows: 30 TR length, 5 TR step
- Group statistics: diagnosis + covariates (age, mean FD; sex/site if available)
- Multiple comparisons:
  - Edge/network metrics: FDR q<0.05
  - Voxelwise ReHo/ISC: voxelwise FDR q<0.05

## Determinism and Logging
- Global random seed from config
- Timestamped logs at `derivatives/logs`
- Reusable manifests under `derivatives/metrics/manifests`

## Notes
- ReHo uses the provided brain mask as GM-mask fallback unless explicit GM masks are supplied.
- ISC is leave-one-out across controls with circular time-shift permutation null, explicitly documented to account for N=4.
