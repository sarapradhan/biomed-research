"""fmri-pipeline: A modular, configuration-driven pipeline for multi-scale functional brain network analysis.

This package provides a complete workflow from fMRIPrep-derivative BOLD data
through multi-scale brain network characterization, including:

- Confound regression with Friston-24 motion parameters, WM/CSF nuisance,
  and optional global signal regression (GSR)
- Motion scrubbing with configurable framewise displacement thresholds
- Schaefer atlas parcellation (100/200/400 ROIs)
- Voxelwise regional homogeneity (ReHo) via Kendall's W
- Static functional connectivity (Pearson + Fisher z-transform)
- Dynamic functional connectivity (sliding-window temporal variability)
- Spatial ICA with cross-subject component matching
- PCA variance decomposition
- Intersubject correlation (ISC) for naturalistic paradigms
- Group-level mass-univariate statistics with FDR correction
- Built-in sensitivity analysis for preprocessing parameter robustness

All parameters are specified in a single YAML configuration file, enabling
zero-code switching between datasets and analysis configurations.

See Also
--------
docs/SOFTWARE_ARCHITECTURE.md : Detailed architectural documentation.
config/pipeline.example.yaml : Annotated configuration template.
"""

__version__ = "0.1.0"
