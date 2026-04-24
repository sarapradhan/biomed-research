---
title: 'fmri-pipeline: A Modular, Configuration-Driven Pipeline for Multi-Scale Functional Brain Network Analysis'
tags:
  - Python
  - fMRI
  - functional connectivity
  - neuroimaging
  - reproducibility
authors:
  - name: Sara Pradhan
    orcid: 0009-0007-8395-6444
    affiliation: 1
affiliations:
  - name: Mountain View High School, Mountain View, CA, USA
    index: 1
date: April 2026
bibliography: paper.bib
---

## Summary

`fmri-pipeline` is a modular, YAML-configured Python package for reproducible analysis of functional magnetic resonance imaging (fMRI) data. The pipeline accepts fMRIPrep [@esteban2019fmriprep] derivatives in BIDS format and produces multi-scale brain network metrics spanning preprocessing quality control, regional homogeneity [@zang2004regional], static and dynamic functional connectivity [@bullmore2009complex; @damaraju2014dynamic], spatial independent component analysis, intersubject correlation [@hasson2004intersubject], and group-level statistics with false discovery rate correction. All parameters—dataset paths, atlas parcellation, analysis toggles, and preprocessing choices—are specified in a single YAML configuration file, enabling zero-code switching between datasets and analysis configurations. A built-in sensitivity analysis module quantifies output robustness across five preprocessing dimensions (global signal regression, atlas resolution, dynamic connectivity window length, motion scrubbing threshold, and spatial smoothing kernel), addressing a widely recognized need for transparent reporting of analytical variability in neuroimaging [@botvinik2020variability].

## Statement of Need

Naturalistic fMRI paradigms, in which participants view movies or listen to narratives during scanning, have emerged as a powerful approach for studying brain function in ecologically valid contexts [@hasson2004intersubject]. Analyzing such data requires coordinating multiple processing stages—confound regression, motion censoring, connectivity estimation, and group comparison—each involving parameter choices that can substantially alter results. The recent "analytical variability" literature has demonstrated that different reasonable preprocessing pipelines applied to the same data can yield divergent conclusions [@botvinik2020variability], underscoring the importance of sensitivity testing and transparent reporting of methodological choices.

The existing tool landscape addresses pieces of this workflow: nilearn [@abraham2014machine] provides flexible connectivity estimation routines, BrainIAK [@tang2023brainiak] specializes in intersubject correlation for naturalistic paradigms, and fMRIPrep [@esteban2019fmriprep] standardizes spatial preprocessing. However, no single package integrates these stages into a configuration-driven workflow that spans from preprocessed BOLD data through multi-scale network characterization, while simultaneously providing built-in sensitivity benchmarking. Researchers—particularly students and early-career investigators—must therefore assemble custom scripts, risking parameter inconsistencies across stages, undocumented analytical choices, and difficulty reproducing or extending prior analyses.

`fmri-pipeline` fills this gap with three design priorities. First, a YAML-driven architecture decouples parameters from code: users switch datasets, atlases, or preprocessing choices by editing a configuration file rather than modifying source code, producing a version-controllable record of every analytical decision. Second, the modular design allows selective execution of individual analysis stages (e.g., connectivity only, or ISC only) while maintaining consistent confound regression and quality control across all modules. Third, the integrated sensitivity analysis module systematically varies key parameters and computes pairwise Spearman rank correlations between the resulting connectivity vectors, producing stability matrices that quantify how robust the pipeline's outputs are to each methodological choice.

## Validation

The pipeline was validated on OpenNeuro ds007318, a publicly available task-based fMRI dataset, with cross-dataset portability demonstrated through configuration files prepared for additional datasets without code modification.

**OpenNeuro ds007318 (task-based fMRI).** A 3-subject, 5-run task paradigm (BIDS task label: "removal") was used for sensitivity benchmarking across 11 parameter conditions. The baseline configuration (GSR on, Schaefer-200 atlas [@schaefer2018local], 30-TR sliding window, FD threshold 0.5 mm, 6 mm smoothing) was compared to systematic perturbations of each dimension. Static functional connectivity vectors (19,900 edges for Schaefer-200, NaN-masked for ROIs outside brain coverage) were compared using Spearman rank correlation (\autoref{fig:sensitivity}).

The results reveal a clear hierarchy of parameter influence. Global signal regression was the only parameter that substantially altered connectivity estimates ($\rho$ = 0.836 relative to baseline), consistent with the well-documented effect of GSR on functional connectivity topology [@power2012spurious; @murphy2017towards]. All other parameters produced near-identical outputs: scrubbing threshold variations yielded $\rho$ $\geq$ 0.997, while dynamic FC window length and spatial smoothing kernel variations produced $\rho$ = 1.000 for static connectivity. Dynamic FC estimates were correspondingly more sensitive to parameter choice ($\rho$ ranging from 0.48 to 1.0), with GSR and window length as the primary sources of variation. Because sliding-window dynamic FC is by construction parameterized by window size, the pipeline presents it as an exploratory robustness module rather than as the basis for state-based inference; users are encouraged to evaluate conclusions across multiple window lengths, as demonstrated here, rather than to rely on a single window choice.

These results support two conclusions relevant to pipeline users: first, the pipeline's outputs are robust to the most commonly varied preprocessing parameters (scrubbing, smoothing, window length), with deviations of at most 0.3%; second, GSR remains the single most consequential methodological choice, reinforcing current best-practice recommendations to report results with and without GSR or to justify the choice explicitly.

**Cross-dataset portability.** To verify that the pipeline generalizes beyond ds007318 without code changes, a YAML configuration was prepared for the CNeuroMod Friends naturalistic movie-viewing dataset. The pipeline's intersubject correlation module implements leave-one-out ISC with circular time-shift permutation testing, and the scene annotation module aligns ISC time courses to external event annotations (e.g., scene boundaries). These modules are included in the test suite and are validated with synthetic data; full empirical validation on CNeuroMod is planned as future work.

![Sensitivity analysis results from OpenNeuro ds007318 (N=3, 5 runs). **A.** Pairwise Spearman rank correlation matrix for static functional connectivity vectors across 9 parameter conditions (Schaefer-200). All conditions except GSR-off show $\rho$ > 0.99. **B.** Parameter impact quantified as mean deviation from baseline (1 − $\rho$). GSR is the dominant source of analytical variability. **C.** Key findings summary.\label{fig:sensitivity}](sensitivity_figures_real/joss_figure_sensitivity_combined.png)

## Software Design

`fmri-pipeline` is organized as a single Python namespace (`fmri_pipeline`) containing 15 analysis and infrastructure modules, orchestrated by a central pipeline runner that reads a YAML configuration file and executes requested analysis stages in dependency order. Seven core analysis modules—preprocessing and quality control, ROI extraction, regional homogeneity, static functional connectivity, dynamic functional connectivity, spatial ICA, and PCA variance decomposition—are supplemented by intersubject correlation, group-level statistics, scene annotation, and sensitivity analysis modules. Supporting modules handle BIDS data ingestion, configuration loading, visualization, and utility functions. Each analysis module reads from and writes to a structured output directory, enabling selective re-execution without repeating upstream stages. Deterministic random seeds are set for all stochastic operations (ICA initialization, permutation sampling) to support reproducibility across runs on the same platform. The test suite (6 test files, 108 tests) validates all core computations using synthetic data, requiring no neuroimaging files to execute.

## Acknowledgements

This work builds on an earlier project conducted through the Lumière Research Scholar Program. The author acknowledges the open-source neuroimaging community, particularly the developers of fMRIPrep, nilearn, and BrainIAK, whose tools and documentation were foundational.

## References
