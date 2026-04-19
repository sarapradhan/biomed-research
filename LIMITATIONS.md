# Limitations

This document describes the known constraints, scope boundaries, and methodological caveats of the current pipeline. It is intended as a transparent acknowledgment of what the pipeline can and cannot do, and where caution is warranted when interpreting results.

## Sample Size

The pipeline has been validated on a pilot sample of N=3 subjects (5 runs) from OpenNeuro ds007318 and a subset of the CNeuroMod Friends dataset (4 subjects). These sample sizes are sufficient to demonstrate pipeline execution and internal consistency, but they are not sufficient for population-level inference. All results from these pilot runs should be treated as hypothesis-generating feasibility demonstrations, not as evidence for biological claims about brain organization.

## Global Signal Regression

The pipeline includes optional global signal regression (GSR), which is a well-known methodological choice with ongoing debate in the field. GSR can reduce global noise contributions and improve specificity of connectivity estimates, but it mathematically introduces negative correlations and can distort group comparisons. The sensitivity analysis module tests the impact of GSR on pipeline outputs. Users should report whether GSR was applied and interpret negative correlations cautiously.

## Parcellation Dependence

Results are computed using the Schaefer parcellation (configurable at 100, 200, or 400 ROIs). Network-level findings may differ across parcellation schemes, and the choice of atlas resolution affects the granularity of connectivity matrices. The sensitivity analysis quantifies this dependence, but users should be aware that results from one atlas resolution do not automatically generalize to others.

## Dynamic FC Window Sensitivity

Dynamic functional connectivity is computed using a sliding-window approach with configurable window length (default: 30 TRs) and step size (5 TRs). Window length trades off temporal resolution against statistical stability of per-window correlation estimates. Very short windows (< 20 TRs) produce noisy FC estimates; very long windows (> 60 TRs) average out genuine temporal dynamics. The sensitivity analysis tests this trade-off explicitly.

## Clinical Data Access

The original project design included a schizophrenia vs. healthy control group comparison using SchizConnect data. SchizConnect was non-functional at the time of development, and this extension was not completed. The group-statistics module exists and is tested, but it has not been exercised on a real clinical cohort with both diagnostic groups.

## ISC Module Scope

The intersubject correlation (ISC) module is implemented with leave-one-out computation and circular time-shift permutation significance testing. It requires naturalistic (shared-stimulus) data with at least 4 subjects watching the same movie. ISC was intentionally omitted from the ds007318 validation because that dataset uses a task paradigm, not a shared naturalistic stimulus. ISC results from the CNeuroMod Friends data should be interpreted as feasibility demonstrations given the small sample.

## Preprocessing Assumptions

The pipeline assumes fMRIPrep derivatives as input. It does not perform raw DICOM-to-NIfTI conversion, fieldmap correction, or cortical surface reconstruction. Users are responsible for running fMRIPrep (or equivalent) before using this pipeline. The confound regression model (Friston-24 + WM/CSF + optional GSR) reflects current best practices but is not the only defensible choice.

## No Deep Learning Integration

The original project plan included deep-learning feature extraction (e.g., CLIP-derived visual features for naturalistic stimuli). This component was not implemented in the current version. Future extensions may add model-based feature comparison, but the current pipeline relies entirely on classical signal processing and statistical methods.

## Reproducibility Caveats

The pipeline uses deterministic random seeds and logs all parameters. However, minor numerical differences may arise across platforms due to floating-point behavior in NumPy/SciPy operations, differences in NiftiMasker implementations across nilearn versions, and OS-level threading behavior. Results should be reproducible within a given environment but may show small (< 1e-6) differences across platforms.

## Scope of Statistical Claims

The group-statistics module uses mass-univariate OLS with FDR correction. This is appropriate for edge-level and voxel-level comparisons but does not account for spatial autocorrelation in voxelwise analyses or network-level multiple comparisons beyond FDR. More sophisticated approaches (e.g., permutation-based cluster correction, network-based statistics) would be needed for publication-quality clinical comparisons.
