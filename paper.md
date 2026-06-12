# A Configuration-Driven Python Pipeline for Reproducible Functional Brain Network Analysis: Internal Validation on a Public fMRI Dataset

**Sara Pradhan**
Mountain View High School, Mountain View, California
pradhan.sara@gmail.com

---

## Abstract

**Background:** Functional magnetic resonance imaging (fMRI) lets researchers estimate coordinated activity across brain regions, but turning raw scanner data into interpretable network metrics requires a chain of analysis decisions — confound regression, parcellation, connectivity estimation, statistical testing — each of which can change the result. This analytical flexibility is a particular obstacle for student investigators, who often assemble custom scripts with little guidance on which choices matter or how to document them.

**Question:** Can a modular, YAML-configured Python pipeline recover established features of large-scale brain organization and produce internally reproducible outputs when applied to a publicly available fMRI dataset?

**Findings:** I validated `fmri-pipeline` on a five-run subset of OpenNeuro ds007318 (3 participants, working-memory task fMRI treated as pseudo-resting-state). The pipeline recovered canonical network organization, confirmed by a permutation test (within-network r̄ = 0.43 vs. between-network r̄ = 0.21, gap = 0.22, *p* < 0.001) and an independent edge-level fingerprint analysis (Spearman rho = 0.32, 95% CI [0.31, 0.34]; network rank order recovered perfectly, rho = 1.00). Regional homogeneity profiles were highly stable within subjects across runs (within-subject r̄ = 0.97). Static functional connectivity showed a within-subject advantage (bootstrap 95% CI on the gap [0.22, 0.35]). A parcellated temporal-ICA proxy was stable across random seeds (20/20 components, mean |*r*| = 0.88). In a sensitivity analysis, global signal regression was the only tested parameter that substantially altered connectivity estimates (Spearman ρ = 0.86 vs. baseline); all other tested parameters produced near-identical outputs (ρ ≥ 0.99).

**Significance:** This is an engineering and reproducibility validation, not a biological-discovery study. On a small public dataset, `fmri-pipeline` produced internally consistent, biologically plausible outputs with complete analytical transparency through a single plain-text configuration file and an automatic validation scorecard, lowering the barrier to reproducible fMRI analysis for student investigators.

---

## Introduction

My interest in brain networks began with watching a family member live with a serious neurological illness, and with how hard it was for the people around her to recognize what was happening until a crisis made it impossible to miss. I did not set out to study her condition, and this project does not: the dataset and sample size here cannot support clinical claims of any kind, and I make none. What stayed with me was a simpler question. When a brain-imaging analysis produces a number, how do we know that number reflects the brain and not just the choices we made along the way? That question is what led me to build and test a transparent, reproducible analysis pipeline rather than simply run an existing one.

Functional MRI allows researchers to estimate coordinated activity across brain regions, but the path from BOLD images to interpretable network results runs through many analysis choices. Reasonable teams can make different decisions about confound regression, motion scrubbing, spatial smoothing, atlas resolution, and statistical modeling, and the "many analysts" literature has shown that this flexibility can produce substantially different conclusions even when teams start from the same dataset [1]. For a student researcher the problem is sharper: a script can run cleanly while still producing outputs that are undocumented, sensitive to arbitrary parameter choices, or hard to reproduce. This raises a concrete, testable question — can a pipeline built around transparent parameterization and built-in internal-validity checks recover established features of large-scale brain organization and produce outputs that are stable across runs and analytical choices?

Several open-source tools already address parts of this workflow. fMRIPrep standardizes functional preprocessing [2], nilearn provides connectivity estimation in Python [3], and BrainIAK offers advanced methods such as intersubject correlation [4]. Integrated environments such as the CONN toolbox [5] and XCP-D [6] provide broader end-to-end processing. These tools are valuable, but they do not remove the need for a project-specific layer that records every parameter choice, links the analysis modules together, and checks whether outputs pass basic internal-validity tests before biological interpretation — and none provides a built-in scorecard designed for a student to confirm pipeline validity before drawing conclusions.

I developed `fmri-pipeline` to fill that gap. All parameters live in a single YAML file, analysis modules run in dependency order, and a built-in reproducibility suite systematically tests whether outputs meet expected internal-consistency criteria. By "multi-scale" I mean analyses spanning the local (voxelwise regional homogeneity) and large-scale network (ROI-level functional connectivity) levels within one workflow. The design goal is not to replace mature neuroimaging packages but to make a complete, student-accessible workflow transparent and self-checking. I tested three hypotheses: (1) the pipeline recovers canonical network organization, measured as stronger within-network than between-network functional connectivity; (2) local and network-level outputs show higher within-subject than between-subject similarity across repeated runs; and (3) static FC estimates are mostly stable across common parameter changes, except for known high-impact choices such as global signal regression.

---

## Materials and Methods

### Dataset

I used OpenNeuro dataset ds007318, a BIDS-formatted fMRI dataset for a working-memory "removal" task in which participants memorized six letters and then removed, updated, or maintained items before an old/new probe judgment [7]. The repository lists 43 participants [7]. This validation used the first three participants and a fixed set of five runs selected for usable coverage during preprocessing:

| Participant | Age | Sex | Group label (participants.tsv) | Runs analyzed |
|---|---:|---|---|---|
| sub-01 | 23 | Female | patient | ses-1, ses-2 |
| sub-02 | 28 | Male | patient | ses-1 |
| sub-03 | 17 | Female | patient | ses-1, ses-2 |

The analyzed sample is therefore **N = 3 participants and 5 BOLD runs** (only a single run from sub-02 was included). This is adequate for a pipeline-validation exercise but not for population-level inference. All three analyzed participants are labeled as patients, so no diagnostic group comparison was performed. Because the validation focuses on within-subject reproducibility and canonical network structure — neither of which requires a control group — the absence of a healthy comparison group does not affect the conclusions reported here, but it does mean these results should not be assumed to generalize to healthy or true resting-state data.

The pipeline used fMRIPrep derivatives [2] as inputs (MNI152NLin2009cAsym space, 2 mm isotropic voxels, TR = 2.0 s, as indicated by the derivative file-naming conventions). Additional scanner acquisition parameters should be confirmed directly from the BIDS JSON sidecars before journal submission. The task BOLD data were treated as pseudo-resting-state: no task-evoked regressors were modeled, and all connectivity analyses (FC, ReHo, ICA proxy) were applied as resting-state-style analyses. This is a reasonable choice for engineering validation but is not equivalent to analyzing true resting-state data.

All data were obtained from the public OpenNeuro repository (OpenNeuro ds007318, snapshot v1.0.0, downloaded February 2026; https://openneuro.org/datasets/ds007318/versions/1.0.0), distributed under a CC0 license. Secondary analysis of this de-identified public dataset did not require new data collection; investigators should nonetheless confirm their own institution's policies for secondary public-data analysis.

### Pipeline overview

`fmri-pipeline` accepts fMRIPrep derivatives in BIDS format and is controlled entirely by a single YAML configuration file. The modules exercised in this validation were: (1) preprocessing quality-control checks and confound regression; (2) regional homogeneity (ReHo), computed voxelwise as Kendall's coefficient of concordance over each voxel's 26-neighbor cluster and summarized to ROI level using the atlas parcellation [10]; (3) static functional connectivity (FC), estimated as pairwise Pearson correlations followed by Fisher-*z* transformation across the Schaefer-200 atlas (200 ROIs, 7-network assignment, 2018 release, applied in MNI152NLin2009cAsym space) [8,9]; (4) exploratory dynamic FC using a sliding-window approach across multiple window lengths [11]; (5) a parcellated temporal-ICA stability check [12,13]; (6) graph-metric summaries of thresholded FC matrices [14]; and (7) a sensitivity-analysis module that systematically varies parameters and computes pairwise Spearman correlations between the resulting connectivity vectors. The pipeline additionally implements intersubject correlation [4] and group-level mass-univariate OLS with Benjamini-Hochberg FDR correction [17]; these modules were not exercised in this single-dataset, single-group validation and are therefore not reported in the Results.

### Confound regression and quality control

Following fMRIPrep preprocessing, confound regression used six rigid-body motion parameters (three translations, three rotations) and their temporal derivatives (12 motion regressors), mean white-matter signal, and mean CSF signal. Volumes with framewise displacement (FD) exceeding 0.5 mm were flagged and excluded from downstream statistical operations via NaN masking. Cleaned BOLD timeseries were bandpass-filtered at 0.01–0.10 Hz with a zero-phase Butterworth filter and spatially smoothed with a 6 mm FWHM Gaussian kernel before connectivity estimation (baseline condition). For the global signal regression (GSR) sensitivity condition, a whole-brain mean BOLD regressor was additionally appended to the confound matrix. GSR was treated as a high-impact parameter because prior work shows it can substantially change FC estimates and network topology [15,16].

### Functional connectivity and the network anchor

For each run the pipeline extracted Schaefer-200 ROI timeseries and computed a 200 × 200 FC matrix; the upper triangle contained 19,900 unique ROI-pair values before masking invalid edges. One ROI in sub-03 ses-2 had a zero-variance timeseries from incomplete brain coverage and was excluded via NaN masking; all statistics used only jointly valid edges. To test canonical organization, edges were divided into within-network (both ROIs in the same Yeo/Schaefer network) and between-network pairs, and mean within- vs. between-network FC was compared with a permutation test (1,000 shuffles of the network-label vector, two-tailed; *p* lower-bounded at 1/1,000). As an independent check, I also computed the Spearman correlation between Fisher-*z* edge values and a same-network membership indicator (3,093 within-network and 16,807 between-network edges).

### Reproducibility analyses

**FC reproducibility.** For each subject I computed the upper-triangle FC vector per run, then within-subject run-pair correlations and between-subject cross-run correlations (Pearson *r* on valid edges, pairwise NaN masking). I compared within- to between-subject similarity with a one-sided Wilcoxon rank-sum test and a bootstrap 95% CI (B = 1,000 run-level resamples) on the within-minus-between gap. Because only sub-01 and sub-03 contributed repeated runs, this run-pair comparison was restricted to those two subjects, yielding 2 within-subject and 4 between-subject run pairs (sub-02's single run cannot form a within-subject pair and is excluded from this test, though it does enter the three-subject ReHo comparison below). With only 2 within-subject and 4 between-subject pairs, this test has theoretical power below 17% even for large true effects, so the bootstrap CI is the primary evidence.

**ReHo stability.** Per run, the ReHo map was computed from cleaned BOLD and summarized to ROI level using the Schaefer-200 atlas; I compared within- to between-subject run-to-run Pearson *r* using the same bootstrap procedure. An early pipeline run wrote empty NIfTI stubs rather than valid ReHo maps because of a bug in the output-writing module (corrected in v1.0.0); I recomputed ReHo from cleaned BOLD using a brain mask derived from temporal variance, which is methodologically equivalent to a gray-matter mask for ROI-level extraction.

**ICA stability.** I ran temporal FastICA [12] on the group-averaged Schaefer-200 ROI timeseries using 5 random seeds (*k* = 20) and matched components across seeds with the Hungarian algorithm [13] on the absolute-correlation matrix; a component was robust if |*r*| ≥ 0.70 against all counterparts. **This is a parcellated temporal-ICA proxy and does not directly validate the pipeline's spatial-ICA module.** A leave-one-run-out cross-validation yielded only 3 run subsets and is reported descriptively only.

**Graph metric stability.** From the group-average FC matrix I computed modularity (Louvain), global efficiency, and clustering coefficient at density thresholds 0.10, 0.15, and 0.20, with bootstrap resampling (B = 500) for 95% CIs and the coefficient of variation.

**Dynamic FC sensitivity.** I computed sliding-window FC variability (SD of the upper-triangle vector over time) at window lengths 20, 30, and 40 TRs, used *k*-means (*k* = 4, k-means++, 10 restarts) to assign states within each condition, and computed the Adjusted Rand Index (ARI) between conditions.

### Sensitivity analysis

I tested output stability across 11 conditions spanning 5 dimensions: GSR (on/off), atlas resolution (Schaefer-100/200/400), dynamic-FC window length (20/30/45/60 TRs; this sweep uses a wider window grid than the 20/30/40-TR reproducibility analysis above), motion scrubbing threshold (FD 0.3/0.5/0.9 mm), and smoothing kernel (4/6/8 mm FWHM). Baseline was Schaefer-200, GSR on, FD 0.5 mm, 6 mm smoothing, 30-TR window. For each condition I compared the static FC upper-triangle vector to baseline with Spearman correlation. Note that static FC does not depend on the dynamic-FC window length, so the window-length conditions are identical to baseline (ρ = 1.00) by construction and are reported only for completeness; the smoothing conditions apply the kernel to the cleaned BOLD before ROI extraction. The figure reports the 9 conditions with atlas resolution held constant (so edges are directly comparable); all 11 are discussed in the text, with the per-condition similarity table (`recommended_defaults.csv`) and the full pairwise similarity matrix (`stability_matrix_static_fc.npy` with `stability_labels.csv`) provided in the archived repository.

### Software and statistics

`fmri-pipeline` is implemented in Python (≥ 3.8) and depends on numpy, scipy, pandas, nilearn, nibabel, statsmodels, scikit-learn, and pyyaml. Group-level testing uses mass-univariate OLS with Benjamini-Hochberg FDR correction [17]. All stochastic operations used deterministic seeds set in the configuration file. The test suite validates core computations on synthetic data and requires no neuroimaging files to run.

---

## Results

### Data quality

All five runs passed the predefined motion-quality criteria (mean FD < 0.5 mm, percent scrubbed < 20%). Mean FD ranged from 0.101 to 0.150 mm, and only sub-02 ses-1 had any scrubbed volumes (≈1.2%) at the FD > 0.5 mm threshold (Table 1). DVARS ranged from 3.68 to 4.41 across runs. One ROI in sub-03 ses-2 had a zero-variance timeseries and was excluded via NaN masking.

**Table 1.** Run-level motion quality-control summary at the FD 0.5 mm scrubbing threshold. All runs passed pipeline exclusion criteria.

| Run | Mean FD (mm) | Median FD (mm) | Scrubbed (%) | DVARS |
|---|---:|---:|---:|---:|
| sub-01, ses-1 | 0.105 | 0.096 | 0.00 | 3.73 |
| sub-01, ses-2 | 0.101 | 0.095 | 0.00 | 3.68 |
| sub-02, ses-1 | 0.144 | 0.119 | 1.18 | 4.41 |
| sub-03, ses-1 | 0.122 | 0.103 | 0.00 | 3.85 |
| sub-03, ses-2 | 0.150 | 0.121 | 0.00 | 3.95 |

### Canonical network organization is recovered

The strongest validation result was the recovery of canonical large-scale network organization. Mean within-network FC (r̄ = 0.43) substantially exceeded mean between-network FC (r̄ = 0.21) across all seven canonical Yeo networks, a gap of 0.22 (*p* < 0.001, permutation test, 1,000 shuffles; *p* at the minimum resolvable value of 1/1,000). Because the Schaefer-200 parcellation assigns ROIs by functional similarity rather than spatial proximity, and the permutation test preserved network-label structure, smoothing-induced proximity correlations are unlikely to explain the effect.

An independent edge-level fingerprint analysis corroborated this result: the Spearman correlation between Fisher-*z* edge values and same-network membership was rho = 0.32 (95% CI [0.31, 0.34]), confirming that same-network ROI pairs are systematically more connected than cross-network pairs across the full edge distribution. Within-network FC was highest for the Visual (Fisher-*z* M = 0.72) and Dorsal Attention (M = 0.49) networks, consistent with strong local synchrony in these systems, and the network rank ordering matched the expected canonical hierarchy perfectly (rank consistency rho = 1.00). Because the data are task fMRI treated as pseudo-resting-state, this is best read as evidence of biologically plausible network structure rather than proof that task-state FC matches resting-state architecture (Figure 1).

![**Figure 1.** Group-average functional connectivity matrix (N = 3 subjects, 5 runs, ds007318). **Panel A:** 200×200 FC heatmap ordered by Yeo-7 network membership (Schaefer-200 parcellation, Fisher-z scale, clipped to [−0.5, 1.5]); axis colour bars denote network identity. The block-diagonal structure reflects canonical large-scale network organization. **Panel B:** Mean within-network FC per canonical network; error bars show standard deviation across the 5 runs. All networks exceed the between-network mean (dashed line; M = 0.21). Vis = Visual; SomMot = Somatomotor; DorsAttn = Dorsal Attention; SalVentAttn = Salience/Ventral Attention; Cont = Frontoparietal/Control; Limbic = Limbic; DMN = Default Mode.](figures/fc_matrix_canonical.png)

### Internal reproducibility analyses

Table 2 summarizes the reproducibility analyses, described below in order of evidential strength.

**Table 2.** Validation scorecard for `fmri-pipeline` on OpenNeuro ds007318 (N = 3 subjects, 5 runs). The FC fingerprint result (rho = 0.32 [0.31, 0.34]) is reported in the text above. CI = bootstrap 95% confidence interval (B = 1,000 run-level resamples). n/a = not assessable at this sample size.

| Validation dimension | Metric | Result | CI (gap) | Status |
|---|---|---|---|:---:|
| Static FC plausibility | Canonical network organization | within r̄ = 0.43, between r̄ = 0.21, gap = 0.22, *p* < 0.001 | — | ✓ |
| ReHo stability | Cross-run spatial similarity | within r̄ = 0.97, between r̄ = 0.51 | [0.436, 0.480] | ✓ |
| ICA stability (temporal proxy) | Component recovery across seeds | 20/20 robust, mean &#124;*r*&#124; = 0.88 | — | ✓ |
| FC reproducibility | Within- vs. between-subject run similarity | within r̄ = 0.71, between r̄ = 0.43, *p* = 0.067 | [0.217, 0.348] | ✓ (CI) |
| Graph metric stability | Bootstrap CI on modularity | *Q* = 0.447, 95% CI [0.391, 0.503], CV = 17% | — | ✓ |
| ICA stability (LORO-CV) | Component recovery across run subsets | N ≤ 3 run subsets | — | n/a |
| Dynamic FC robustness | Multi-window ARI | ARI 0.42–0.57 | — | n/a |

**ReHo stability** was the strongest reproducibility result. Within-subject run-to-run Pearson *r* for ROI ReHo profiles was 0.97 versus 0.51 between subjects (gap = 0.455, 95% CI [0.436, 0.480]). The near-perfect within-subject similarity indicates the ReHo spatial pattern is highly consistent across runs, and the large gap shows individual differences are preserved. The corresponding standardized effect size is very large (Cohen's *d* = 31.2); this reflects the contrast between near-ceiling within-subject similarity and moderate between-subject similarity in a small sample and is reported with that caveat rather than as a precise reliability estimate.

**ICA stability** across random seeds was strong: all 20 components were robustly recovered (|*r*| ≥ 0.70) across all 5 seeds, mean |*r*| = 0.88, indicating the temporal decomposition is not sensitive to initialization. This validates seed stability of the parcellated temporal-ICA proxy only, not the spatial-ICA module. The LORO-CV (3 run subsets) is reported descriptively: mean |*r*| = 0.411, consistent with the known instability of temporal ICA on very small datasets.

**FC reproducibility** showed a consistent within-subject advantage: within-subject run-pair FC correlation r̄ = 0.71 versus between-subject r̄ = 0.43 (gap = 0.28). The bootstrap 95% CI on the gap [0.217, 0.348] entirely excludes zero. The Wilcoxon rank-sum test did not reach significance (*p* = 0.067); given only 2 within-subject pairs the test is severely underpowered, so this should be read as preliminary internal-reproducibility evidence, not definitive test-retest reliability, and the non-significant *p* should not be taken as a null result.

**Graph metric stability** was adequate. Bootstrap resampling (B = 500) of modularity at density 0.15 gave *Q* = 0.447 (95% CI [0.391, 0.503], CV = 17%). No published consensus CV threshold exists for this context; 17% is reported descriptively. The observed *Q* falls within the range commonly reported for resting-state networks at comparable atlas resolutions [14], providing an external anchor for the observed network separation.

### Sensitivity to preprocessing parameters

Global signal regression was the only parameter that substantially altered connectivity estimates (Spearman ρ = 0.862 relative to baseline), consistent with the established literature on GSR's effect on FC topology [15,16]. All other tested parameters produced near-identical static FC outputs when atlas resolution was held constant: smoothing-kernel variations were very close to baseline (4 mm: ρ = 0.997; 8 mm: ρ = 0.997), window-length variations were identical to baseline by construction (ρ = 1.00, as static FC is invariant to the dynamic-FC window), and scrubbing-threshold variations remained very high (FD 0.3 mm: ρ = 0.991; FD 0.9 mm: ρ = 0.997). With the exception of GSR, the pipeline's static FC outputs were robust (ρ ≥ 0.99) across the most commonly varied preprocessing parameters in this low-motion subset (Figure 2).

Dynamic FC variability (upper-triangle SD over time) decreased monotonically with window size (*W* = 20 TR: 0.409; 30 TR: 0.308; 40 TR: 0.253), as expected from the averaging properties of longer windows. State-assignment consistency across window lengths was moderate (ARI 0.42–0.57), reflecting the dependence of sliding-window clustering on temporal resolution. Dynamic FC outputs are included for exploratory characterization only and are not the basis for state-based inference at this sample size.

![**Figure 2.** Sensitivity analysis from ds007318 (N = 3, 5 runs). **A.** Pairwise Spearman correlation matrix for static FC vectors across 9 parameter conditions (Schaefer-200, varying GSR, scrubbing threshold, window length, smoothing kernel); all conditions except GSR-off produce ρ > 0.99 (window-length conditions are identical to baseline by construction, as static FC does not depend on the dynamic-FC window). **B.** Parameter impact as mean deviation from baseline (1 − ρ); GSR is the dominant source of analytical variability. **C.** Summary of key findings.](sensitivity_figures_real/joss_figure_sensitivity_combined.png)

---

## Discussion

### Principal findings

`fmri-pipeline` produced coherent, internally reproducible functional-network outputs from a small public fMRI dataset without modification of source code — only the YAML configuration was changed to point at the new data. The strongest result, recovery of canonical network organization confirmed by both a permutation test (*p* < 0.001) and an independent edge-level fingerprint (rho = 0.32, rank consistency 1.00), establishes that the pipeline's FC matrices are biologically plausible: within-network connections are systematically stronger than between-network connections across all seven Yeo networks. This is the expected signature of large-scale brain modularity [14], and its recovery is consistent with correct functioning of the atlas-parcellation and connectivity-estimation steps.

The reproducibility analyses collectively indicate the outputs are not arbitrary. ReHo profiles were nearly perfectly reproduced within subjects across runs; static FC showed a within-subject advantage with a bootstrap CI that excludes zero; and the parcellated temporal-ICA proxy was stable across seeds. The paper's strongest defensible claim is that the pipeline produced biologically plausible and internally reproducible outputs on a small public subset, supporting its use as a transparent validation workflow for future, larger analyses. The current data do **not** support claims of disease-biomarker discovery, schizophrenia-related dysconnectivity, or general test-retest reliability, and I do not make them.

### Why the network-anchor result matters

The network-anchor test functions like a diagnostic check on the whole FC workflow. If the atlas labels, timeseries extraction, confound regression, or FC computation were seriously misconfigured, the matrix would be unlikely to show stronger within- than between-network connectivity. Recovering the expected pattern does not prove every module is correct, but it provides meaningful evidence that the core workflow is functioning sensibly — much as checking that a new thermometer reads a plausible room temperature before an experiment does not answer the scientific question but does rule out using a broken instrument.

### Sensitivity to parameter choices

The sensitivity analysis directly addresses hypothesis 3 and supports it for every parameter except GSR. Crucially, the pipeline does not eliminate analytical variability — it documents it. The GSR result (ρ = 0.86) shows that a single parameter can substantially change connectivity topology; the pipeline's contribution is to make that choice explicit, version-controlled, and reproducible in a plain-text file rather than buried in custom scripts. Practically, minor differences in scrubbing threshold or smoothing kernel are unlikely to alter static FC conclusions in low-motion data, whereas GSR usage should be matched whenever cross-study comparisons are made and treated as a pre-specified decision rather than a casual toggle.

### Limitations

Several limitations constrain these conclusions. First, the validation sample was very small — three participants and five runs, with only two participants contributing repeated runs — which severely limits power for between-subject comparisons; the non-significant FC reproducibility test reflects sample size, not absence of effect. Second, all analyzed participants were labeled as patients, so no healthy-control comparison was possible. Third, the data were acquired during a working-memory task and treated as pseudo-resting-state; this is useful for engineering validation but is not equivalent to true resting-state data. Fourth, the ICA stability analysis used a parcellated temporal-ICA proxy and does not validate the spatial-ICA module. Fifth, dynamic FC analyses are exploratory at this sample size. Sixth, the ReHo recomputation used a temporal-variance brain mask rather than a gray-matter segmentation mask; for ROI-level extraction this distinction is negligible but is noted for completeness.

### Implications and next steps

`fmri-pipeline` is designed to lower the barrier to reproducible fMRI analysis for student investigators. Because all analytical decisions live in a version-controlled plain-text file, the methodology can be shared alongside results, and the automatic reproducibility scorecard lets a student check internal consistency before drawing biological conclusions. The clear next step is to run the same validation framework on a larger dataset with more repeated sessions — ideally ≥ 20 participants, ≥ 2 usable runs each, a true resting-state or naturalistic-viewing design, direct validation of the spatial-ICA module, and pre-registered settings for GSR, scrubbing, smoothing, and atlas resolution. A configuration for a naturalistic-viewing dataset has been prepared, and running it is the highest-impact way to move this work from an engineering-validation paper toward a stronger reproducibility methods paper.

---

## Conclusion

`fmri-pipeline` generated biologically plausible and internally consistent functional-network outputs from a small public fMRI subset, with the strongest results being recovery of canonical network organization, high within-subject ReHo stability, a preliminary within-subject FC advantage, and predictable sensitivity to global signal regression. The manuscript should be read as a reproducible pipeline-validation study rather than a biological-discovery study. With larger repeated-session validation and public archiving of all code and result files, this work could provide a useful, student-accessible framework for transparent fMRI network analysis.

---

## Data and Code Availability

The input fMRI dataset is publicly available as OpenNeuro ds007318 (https://openneuro.org/datasets/ds007318) [7], distributed under a CC0 license. The source code for `fmri-pipeline` is openly available at https://github.com/sarapradhan/biomed-research (release v1.0.1, MIT License) and is archived at https://doi.org/10.5281/zenodo.20280072. The pipeline configuration files, reproducibility configuration, QC table, and sensitivity outputs (`recommended_defaults.csv`, `stability_matrix_static_fc.npy`, `stability_labels.csv`, `cross_atlas_summary.csv`) used to generate the reported results are included in the archived repository.

---

## Acknowledgements

This work was made possible by the open-source neuroimaging community. The author thanks the developers of fMRIPrep, nilearn, nibabel, scipy, scikit-learn, statsmodels, and the Schaefer-200 atlas, whose freely available tools form the foundation of `fmri-pipeline`. The author also thanks the creators of OpenNeuro and the team that collected and shared dataset ds007318, as well as the study participants whose data made this validation possible.

The author thanks Arefeh Sherafati, Research Scientist at St. Jude Children's Research Hospital, for guidance and feedback throughout this project.

The author used generative AI (Anthropic's Claude) as a writing and coding assistant during manuscript preparation and software development. All scientific content, analyses, interpretations, and final code were designed, reviewed, and verified by the author, who takes full responsibility for the work.

---

## References

[1] Botvinik-Nezer, R., Holzmeister, F., Camerer, C. F., et al. (2020). Variability in the analysis of a single neuroimaging dataset by many teams. *Nature, 582*, 84–88. https://doi.org/10.1038/s41586-020-2314-9

[2] Esteban, O., Markiewicz, C. J., Blair, R. W., et al. (2019). fMRIPrep: a robust preprocessing pipeline for functional MRI. *Nature Methods, 16*, 111–116. https://doi.org/10.1038/s41592-018-0235-4

[3] Abraham, A., Pedregosa, F., Eickenberg, M., et al. (2014). Machine learning for neuroimaging with scikit-learn. *Frontiers in Neuroinformatics, 8*, 14. https://doi.org/10.3389/fninf.2014.00014

[4] Kumar, M., Anderson, M. J., Antony, J. W., et al. (2021). BrainIAK: The Brain Imaging Analysis Kit. *Aperture Neuro, 1*, 1–19. https://doi.org/10.52294/31bb5b68-2184-411b-8c00-a1dacb61e1da

[5] Whitfield-Gabrieli, S., & Nieto-Castanon, A. (2012). Conn: a functional connectivity toolbox for correlated and anticorrelated brain networks. *Brain Connectivity, 2*(3), 125–141. https://doi.org/10.1089/brain.2012.0073

[6] Mehta, K., Salo, T., Madison, T. J., et al. (2024). XCP-D: A robust pipeline for the post-processing of fMRI data. *Imaging Neuroscience, 2*, 1–26. https://doi.org/10.1162/imag_a_00257

[7] Wang, T., Li, Y., & Zhao, X. (2026). *fMRI Study Dataset* [Data set]. OpenNeuro. https://doi.org/10.18112/openneuro.ds007318.v1.0.0

[8] Schaefer, A., Kong, R., Gordon, E. M., et al. (2018). Local-global parcellation of the human cerebral cortex from intrinsic functional connectivity MRI. *Cerebral Cortex, 28*(9), 3095–3114. https://doi.org/10.1093/cercor/bhx179

[9] Yeo, B. T. T., Krienen, F. M., Sepulcre, J., et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *Journal of Neurophysiology, 106*(3), 1125–1165. https://doi.org/10.1152/jn.00338.2011

[10] Zang, Y., Jiang, T., Lu, Y., He, Y., & Tian, L. (2004). Regional homogeneity approach to fMRI data analysis. *NeuroImage, 22*(1), 394–400. https://doi.org/10.1016/j.neuroimage.2003.12.030

[11] Damaraju, E., Allen, E. A., Belger, A., et al. (2014). Dynamic functional connectivity analysis reveals transient states of dysconnectivity in schizophrenia. *NeuroImage: Clinical, 5*, 298–308. https://doi.org/10.1016/j.nicl.2014.07.003

[12] Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis. *IEEE Transactions on Neural Networks, 10*(3), 626–634. https://doi.org/10.1109/72.761722

[13] Kuhn, H. W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly, 2*(1–2), 83–97. https://doi.org/10.1002/nav.3800020109

[14] Bullmore, E., & Sporns, O. (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nature Reviews Neuroscience, 10*, 186–198. https://doi.org/10.1038/nrn2575

[15] Power, J. D., Barnes, K. A., Snyder, A. Z., Schlaggar, B. L., & Petersen, S. E. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage, 59*(3), 2142–2154. https://doi.org/10.1016/j.neuroimage.2011.10.018

[16] Murphy, K., & Fox, M. D. (2017). Towards a consensus regarding global signal regression for resting state functional connectivity MRI. *NeuroImage, 154*, 169–173. https://doi.org/10.1016/j.neuroimage.2016.11.052

[17] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B, 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
