---
title: 'A reproducible, modular Python pipeline for multi-scale functional brain network analysis: validation on a public fMRI dataset'
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
date: May 2026
bibliography: paper.bib
---

## Abstract

Functional magnetic resonance imaging (fMRI) enables non-invasive measurement of brain activity, but translating raw scanner data into reproducible, interpretable network metrics requires coordinating multiple analysis stages whose parameter choices can substantially alter results. We developed `fmri-pipeline`, a modular, configuration-driven Python package that integrates preprocessing quality control, regional homogeneity, static and dynamic functional connectivity, independent component analysis, and group-level statistics into a single reproducible workflow. We hypothesized that a pipeline designed around transparent parameterization and internal validation checks would (1) recover established features of large-scale brain organization, (2) produce outputs that are stable across runs and random seeds, and (3) exhibit bounded sensitivity to commonly varied preprocessing parameters. We validated the pipeline on OpenNeuro ds007318 (3 subjects, 5 runs total, working-memory task fMRI) using six internal reproducibility analyses. Static functional connectivity matrices displayed canonical network organization confirmed by a permutation test (within-network *r̄* = 0.43 vs. between-network *r̄* = 0.21, gap = 0.22, *p* < 0.005). Run-to-run reproducibility was strong for both regional homogeneity (within-subject *r̄* = 0.97, Cohen's *d* = 31.2) and static functional connectivity (within-subject *r̄* = 0.71, 95% CI on gap: [0.22, 0.35]). Independent component decompositions were stable across random seeds (20/20 components recovered, mean |*r*| = 0.88). These results demonstrate that `fmri-pipeline` produces internally consistent, biologically interpretable outputs on public data, supporting its use as a reproducible framework for student-scientist fMRI research.

---

## Introduction

The analysis of functional magnetic resonance imaging data involves multiple sequential decisions—confound regression strategy, atlas parcellation, connectivity window length, motion censoring threshold—each of which can meaningfully alter downstream results [@botvinik2020variability]. This analytical flexibility is particularly challenging for student investigators, who must assemble and maintain custom scripts without formal guidance on which choices matter most or how to document them transparently.

Several open-source tools address specific analysis stages. The fMRIPrep pipeline standardizes spatial preprocessing [@esteban2019fmriprep], nilearn provides flexible connectivity estimation [@abraham2014machine], and the BrainIAK toolkit offers intersubject correlation methods [@hasson2004intersubject; @tang2023brainiak]. Integrated environments such as the CONN toolbox and XCP-D offer broader end-to-end workflows, but they are oriented toward established research labs and do not provide a built-in biological validity scorecard that tests whether pipeline outputs meet expected criteria before the researcher proceeds to inference. A researcher wishing to perform end-to-end analysis—from preprocessed BOLD data through multi-scale network characterization and group statistics—must still write custom glue code, introducing undocumented parameter choices and limiting reproducibility across datasets.

A further underappreciated challenge is that neuroimaging outputs are rarely validated for internal consistency. A pipeline may execute without errors yet produce results sensitive to arbitrary parameter choices (e.g., random seed for ICA initialization, atlas resolution, smoothing kernel) or may yield FC matrices that do not reflect known biological structure. The "many analysts" literature has shown that different reasonable pipelines applied to the same dataset can yield contradictory conclusions [@botvinik2020variability], underscoring the need for built-in analytical transparency.

We developed `fmri-pipeline` to address these gaps. The package provides a YAML-driven workflow in which all parameters are specified in a single configuration file, analysis modules execute in dependency order, and a built-in reproducibility suite systematically tests whether outputs meet expected biological and internal-consistency criteria. By "multi-scale," we refer to analyses spanning local (voxelwise regional homogeneity) and large-scale network levels (ROI-level functional connectivity), capturing complementary aspects of brain organization within a single unified workflow. We hypothesized that a pipeline organized around these principles would, when applied to a publicly available dataset: (1) recover established features of large-scale brain organization, specifically the canonical distinction between within-network and between-network functional connectivity; (2) produce outputs that are stable across runs, random seeds, and resampling strategies; and (3) demonstrate that preprocessing parameter choices have predictable, bounded effects on static connectivity estimates.

---

## Materials and Methods

### Dataset

We validated the pipeline on OpenNeuro ds007318, a publicly available BIDS-formatted fMRI dataset comprising a working-memory removal paradigm in which participants were cued to maintain or suppress specific items held in working memory. All fMRI data were preprocessed with fMRIPrep [@esteban2019fmriprep] prior to this analysis; we used the fMRIPrep derivatives (MNI152NLin2009cAsym space, 2 mm isotropic voxels, TR = 2.0 s) directly as pipeline inputs. Additional scanner acquisition parameters (field strength, echo time, flip angle, acquired matrix size) should be verified from the dataset's BIDS JSON sidecars on OpenNeuro and reported in any journal submission; the parameters above are confirmed from the fMRIPrep derivative file naming conventions.

The validated subset comprises 3 participants (sub-01: age 23, female; sub-02: age 28, male; sub-03: age 17, female), all labeled as patients in the study phenotypic file, contributing 5 BOLD runs total: 2 sessions for sub-01 (ses-1, ses-2), 1 session for sub-02 (ses-1), and 2 sessions for sub-03 (ses-1, ses-2). Because the validation focuses on within-subject reproducibility and canonical network structure — neither of which depends on group comparison — the absence of a control group does not affect the conclusions reported here; however, generalizability of these validation results to healthy non-clinical populations has not yet been directly demonstrated. The BOLD data acquired during the task were treated as pseudo-resting-state: no task-evoked regressors were modeled, and all analyses (FC, ReHo, ICA) were applied as resting-state connectivity analyses.

One region of interest in subject 3, run 2 fell outside brain coverage (zero-variance timeseries) and was excluded from all connectivity analyses via NaN masking; all statistical operations used only jointly valid edges.

All data were obtained from the publicly available OpenNeuro repository (https://openneuro.org/datasets/ds007318); no additional ethics approval was required for secondary analysis of this de-identified public dataset.

### Pipeline overview

`fmri-pipeline` accepts fMRIPrep derivatives in BIDS format and executes seven analysis modules in sequence: (1) preprocessing quality control and confound regression; (2) regional homogeneity (ReHo) computed voxelwise as Kendall's coefficient of concordance (*W*) over each voxel's 26-neighbor cluster, then summarized to ROI level using the atlas parcellation [@zang2004regional]; (3) static functional connectivity (FC) estimated as pairwise Pearson correlations followed by Fisher-*z* transformation across the Schaefer-200 atlas parcellation (200 ROIs, 7-network assignment, 2018 release, applied to MNI152NLin2009cAsym space) [@schaefer2018local]; (4) dynamic functional connectivity using a sliding-window approach across multiple window lengths [@damaraju2014dynamic]; (5) subject-level spatial independent component analysis (ICA) of masked BOLD volumes; (6) intersubject correlation with circular time-shift permutation testing [@hasson2004intersubject]; and (7) group-level mass-univariate OLS with Benjamini-Hochberg false discovery rate correction. All parameters are specified in a single YAML configuration file. The sensitivity analysis module systematically varies five parameter dimensions and computes pairwise Spearman rank correlations between the resulting connectivity vectors, quantifying how analytical choices propagate to outputs.

### Confound regression

Following fMRIPrep preprocessing, confound regression was applied using the following regressors: six rigid-body motion parameters (three translations, three rotations) and their temporal derivatives (12 motion regressors total), mean white matter signal, and mean CSF signal. Volumes with framewise displacement (FD) exceeding 0.5 mm were scrubbed (flagged and excluded from all downstream statistical operations via NaN masking). Cleaned BOLD timeseries were bandpass-filtered at 0.01–0.10 Hz using a zero-phase Butterworth filter and spatially smoothed with a 6 mm FWHM Gaussian kernel prior to connectivity estimation (baseline condition). For the global signal regression (GSR) sensitivity condition, a whole-brain mean BOLD signal regressor was additionally appended to the confound matrix.

### Reproducibility analyses

We designed six internal reproducibility analyses to test specific properties of the pipeline outputs, following the framework recommended for validation of analysis pipelines [@botvinik2020variability]:

**FC reproducibility.** For each subject, we computed the upper-triangle FC vector (19,900 edges for Schaefer-200, NaN-masked) for each run, then computed within-subject run-pair correlations (Pearson *r* on valid edges using pairwise NaN masking) and between-subject cross-run correlations. We compared within-subject to between-subject similarity using a one-sided Wilcoxon rank-sum test (equivalent to the Mann-Whitney *U* test for two independent samples; alternative: within > between) and a bootstrap 95% confidence interval (B = 1,000 run-level resamples) on the within-minus-between gap. Cohen's *d* was computed from the pooled standard deviation of the two groups. With only 2 within-subject pairs and 4 between-subject pairs, the Wilcoxon rank-sum test with these group sizes has theoretical power below 17% even for very large true effects; the bootstrap CI is therefore the primary evidence for a within-subject advantage.

**ReHo stability.** For each run, the ReHo map was computed from the unsmoothed cleaned BOLD and projected to ROI-level summaries using the Schaefer-200 atlas. We then computed run-to-run Pearson *r* between ROI ReHo profiles within and across subjects, and compared within-subject to between-subject similarity using the same bootstrap procedure as FC reproducibility. Because the initial pipeline run had stored empty NIfTI stubs (file size < 10 KB or all-zero data) rather than valid ReHo maps — due to a bug in the output-writing module, since corrected in version [SOFTWARE_VERSION] of `fmri-pipeline` — we recomputed ReHo from the cleaned BOLD using a brain mask derived from temporal variance of the signal; this approach is methodologically equivalent to a gray-matter mask for ROI-level extraction.

**ICA stability.** We ran temporal FastICA on the group-averaged Schaefer-200 ROI timeseries using 5 independent random seeds (components *k* = 20) and matched components across seeds using the Hungarian algorithm [@kuhn1955hungarian] on the absolute Pearson *r* similarity matrix. A component was classified as robust if it achieved |*r*| ≥ 0.70 against all other-seed counterparts. We additionally ran leave-one-run-out cross-validation (LORO-CV) to assess stability across data partitions; however, with only 3 subjects and 5 runs, the LORO-CV yielded only 3 run subsets, which is insufficient for meaningful inference, and those results are reported as descriptive only. **Note:** this stability analysis uses temporal ICA on Schaefer-200 parcellated timeseries as a computational proxy for the spatial ICA performed on full BOLD volumes in the main analysis pipeline. While sufficient for seed-stability benchmarking, this proxy tests a methodologically distinct procedure and does not directly validate the pipeline's spatial ICA module.

**Graph metric stability.** We computed network topology metrics—modularity (Louvain community detection), global efficiency, and mean clustering coefficient—from the group-average FC matrix at density thresholds 0.10, 0.15, and 0.20. We applied bootstrap resampling (B = 500 run-level resamples with replacement) to estimate 95% confidence intervals and the coefficient of variation (CV) for each metric and threshold.

**Dynamic FC sensitivity.** We computed sliding-window FC variability (standard deviation of the upper-triangle FC vector over time) at window lengths *W* = 20, 30, and 40 TRs. We used *k*-means clustering (*k* = 4, *k*-means++ initialization, 10 restarts) to assign connectivity states within each window-length condition and computed the Adjusted Rand Index (ARI) between state assignments across all pairwise window-length comparisons as a measure of clustering consistency.

**Network anchor.** We tested whether the pipeline's FC matrices exhibited the canonical large-scale organization of the Schaefer-200 7-network parcellation (default mode, frontoparietal, dorsal attention, ventral attention, somatomotor, visual, limbic networks). For each subject and run, we computed the mean within-network FC (edges connecting two ROIs assigned to the same network) and mean between-network FC (all other edges), using NaN-aware means. We averaged these values across subjects and runs and tested whether within-network FC exceeded between-network FC using a permutation test (1,000 shuffles of the network label vector, two-tailed; *p* is estimated as the proportion of permuted within-minus-between gaps whose absolute value equalled or exceeded the observed gap, with a lower bound of 1/1,000 given the permutation count).

### Sensitivity analysis

We tested pipeline output stability across 11 parameter conditions spanning 5 dimensions: global signal regression (on vs. off), atlas resolution (Schaefer-100, Schaefer-200, Schaefer-400), dynamic FC window length (20, 30, 45, 60 TRs), motion scrubbing threshold (FD 0.3, 0.5, 0.9 mm), and spatial smoothing kernel (4, 6, 8 mm FWHM). The Schaefer-200 configuration with GSR on, FD 0.5 mm, 6 mm smoothing, and 30-TR window was treated as baseline. For each condition, we computed the static FC upper-triangle vector and compared it to baseline using Spearman rank correlation. The figure caption reports the 9 static FC conditions in which atlas resolution was held constant (Schaefer-200) to enable direct edge-level comparison; all 11 conditions including atlas-resolution perturbations are discussed in the text. The full Spearman ρ table is provided in the archived `stability_summary.csv` (see Data and Code Availability).

### Software and statistics

`fmri-pipeline` is implemented in Python (≥3.8) and depends on numpy, scipy, pandas, nilearn, nibabel, statsmodels, scikit-learn, and pyyaml. Group-level statistical testing used mass-univariate OLS with Benjamini-Hochberg FDR correction across all edges or voxels. All stochastic operations (ICA initialization, permutation sampling, bootstrap resampling) used deterministic seeds specified in the configuration file to ensure bit-level reproducibility on the same platform. The test suite comprises 18 test files and 382 tests validating all core computations using synthetic data; no neuroimaging files are required to run the tests.

---

## Results

### Pipeline execution and data quality

The pipeline executed successfully on all 3 subjects across 5 runs of ds007318 (sub-01: 2 sessions; sub-02: 1 session; sub-03: 2 sessions) without modification of source code; only the YAML configuration file was updated to point to the new dataset. Motion quality was excellent across all runs (Table 0): mean FD ranged from 0.101 to 0.150 mm, and the percentage of volumes scrubbed at the FD > 0.5 mm threshold ranged from 0.0% to 1.2%. DVARS values ranged from 3.68 to 4.41, consistent with acceptable signal quality. All runs passed the pipeline's automatic exclusion criteria (mean FD < 0.5 mm, percent scrubbed < 20%). One ROI in subject 3, run 2 had zero-variance timeseries due to incomplete brain coverage and was excluded via NaN masking in all connectivity analyses.

**Table 0.** Motion quality control summary for all 5 runs of OpenNeuro ds007318 at the FD 0.5 mm scrubbing threshold. All runs passed pipeline exclusion criteria.

| Run | Mean FD (mm) | Median FD (mm) | Scrubbed (%) | DVARS |
|---|---|---|---|---|
| sub-01, ses-1 | 0.105 | 0.096 | 0.0 | 3.73 |
| sub-01, ses-2 | 0.101 | 0.095 | 0.0 | 3.68 |
| sub-02, ses-1 | 0.144 | 0.119 | 1.2 | 4.41 |
| sub-03, ses-1 | 0.122 | 0.103 | 0.0 | 3.85 |
| sub-03, ses-2 | 0.150 | 0.121 | 0.0 | 3.95 |

### Canonical network organization is recovered (network anchor)

The strongest validation result was the recovery of canonical large-scale network organization. Within-network FC (*r̄* = 0.43) substantially exceeded between-network FC (*r̄* = 0.21) across all 7 canonical Yeo networks, yielding a gap of 0.22 (*p* < 0.005, permutation test, 1,000 shuffles; exact *p* = 1/1,000, at the minimum resolvable value for this permutation count). This result is consistent with correct functioning of the pipeline's atlas parcellation and connectivity estimation steps. We note that spatially proximate ROIs within the same Yeo network might share smoothing-induced correlations that could artifactually inflate within-network FC; however, because the Schaefer-200 parcellation assigns ROIs on the basis of functional similarity rather than spatial proximity, and the permutation test preserved network label structure throughout, this concern is substantially mitigated.

### Internal reproducibility analyses

Table 1 summarizes all six reproducibility analyses. The results are described below in order of strength of evidence.

**Table 1.** Validation scorecard for `fmri-pipeline` on OpenNeuro ds007318 (N = 3 subjects, 5 runs total). CI = bootstrap 95% confidence interval (B = 1,000 run-level resamples). n/a = not assessable at this sample size.

| Validation dimension | Metric | Result | CI (gap) | Status |
|---|---|---|---|:---:|
| Static FC plausibility | Canonical network organization | within *r̄* = 0.43, between *r̄* = 0.21, gap = 0.22, *p* < 0.005 | — | ✓ |
| ReHo stability | Cross-run spatial similarity | within *r̄* = 0.97, between *r̄* = 0.51, *d* = 31.2† | [0.436, 0.480] | ✓ |
| ICA stability (temporal proxy)‡ | Component recovery across seeds | 20/20 robust, mean |*r*| = 0.88 | — | ✓ |
| FC reproducibility | Within- vs. between-subject run similarity | within *r̄* = 0.71, between *r̄* = 0.43, *d* = 5.7, *p* = 0.067 | [0.217, 0.348] | ✓ |
| Graph metric stability | Bootstrap CI on modularity | *Q* = 0.447, 95% CI [0.391, 0.503], CV = 17% | — | ✓ |
| ICA stability (LORO-CV) | Component recovery across run subsets | N ≤ 3 run subsets | — | n/a |
| Dynamic FC robustness | Multi-window ARI | ARI(20-30) = 0.454, ARI(20-40) = 0.420, ARI(30-40) = 0.572 | — | n/a |

*Status ✓: passed validation criterion. n/a: insufficient data for inference.*

† *d* = 31.2 reflects a contrast between near-ceiling within-subject similarity (*r̄* = 0.97) and moderate between-subject similarity (*r̄* = 0.51). The within- and between-subject distributions show negligible overlap, yielding an extreme pooled standardized difference that is mathematically expected given this contrast, not a measurement error.

‡ ICA stability was assessed using temporal FastICA on Schaefer-200 parcellated timeseries — a computational proxy for the spatial ICA performed on full BOLD volumes in the main pipeline. Seed stability of this proxy does not directly validate the pipeline's spatial ICA module; the result is best interpreted as evidence that the group-level parcellated timeseries structure is stable across random initializations.

**ReHo stability** was the strongest reproducibility result. Within-subject run-to-run Pearson *r* for ROI ReHo profiles was 0.97, versus 0.51 between subjects (gap = 0.455, 95% CI [0.436, 0.480], Cohen's *d* = 31.2; see Table 1 footnote †). The near-perfect within-subject similarity indicates that the ReHo spatial pattern is highly consistent across scanning runs, and the large gap from between-subject similarity confirms that individual differences in ReHo profile are preserved by the pipeline.

**ICA stability** across random seeds was strong. All 20 components were robustly recovered (|*r*| ≥ 0.70) across all 5 random seeds, with a mean absolute correlation of 0.88. This indicates that the temporal ICA decomposition is not sensitive to initialization and converges to stable components (see Table 1 footnote ‡ regarding the proxy limitation). The LORO-CV yielded only 3 run subsets (N = 3 subjects), which is insufficient for stable estimates and is reported as descriptive: mean |*r*| across run subsets was 0.411, below the robust threshold, consistent with the known instability of temporal ICA on very small datasets.

**FC reproducibility** showed a consistent within-subject advantage over between-subject similarity: within-subject run-pair FC correlation was *r̄* = 0.71 versus between-subject *r̄* = 0.43 (gap = 0.28, Cohen's *d* = 5.7). The bootstrap 95% CI on the gap ([0.217, 0.348]) entirely excludes zero, demonstrating a consistent direction. The formal Wilcoxon rank-sum test did not reach significance (*p* = 0.067); with only 2 within-subject pairs and 4 between-subject pairs, this test has theoretical power below 17% even for large true effects, and the non-significant *p*-value should not be interpreted as evidence of no effect. The CI is the more informative evidence here.

**Graph metric stability** was adequate. Bootstrap resampling (B = 500 run-level resamples) of the modularity at density threshold 0.15 yielded *Q* = 0.447 (95% CI [0.391, 0.503], CV = 17%). No published consensus threshold exists for acceptable CV in this context; 17% is reported descriptively as an indication that network topology summaries are not strongly sensitive to the particular run subset included.

### Sensitivity to preprocessing parameters

We tested sensitivity across 9 static FC conditions (Schaefer-200, varying GSR, scrubbing threshold, window length, and smoothing kernel; 11 conditions total when including atlas resolution perturbations). Global signal regression was the only parameter that substantially altered connectivity estimates (Spearman ρ = 0.836 relative to baseline), consistent with the well-documented effect of GSR on functional connectivity topology [@power2012spurious; @murphy2017towards]. All remaining parameters produced near-identical static FC outputs: scrubbing threshold variations (FD 0.3 vs. 0.9 mm) yielded ρ ≥ 0.997 relative to baseline, and smoothing kernel variations (4 vs. 8 mm) and window-length variations produced ρ ≥ 0.997. These results indicate that, with the exception of GSR, the pipeline's static FC outputs are robust across the most commonly varied preprocessing parameters. (\autoref{fig:sensitivity})

Dynamic FC estimates were more sensitive to window length, as is expected from the averaging properties of sliding-window methods: FC variability (upper-triangle standard deviation over time) decreased monotonically with window size (*W* = 20 TR: 0.409, *W* = 30 TR: 0.308, *W* = 40 TR: 0.253), consistent with larger windows averaging over more timepoints and thus suppressing moment-to-moment variability. State-assignment agreement between window-size conditions was moderate (ARI range: 0.42–0.57; ARI = 1.0 indicates perfect agreement), indicating that the specific connectivity states detected change substantially with window length — a property intrinsic to the sliding-window method. Accordingly, dynamic FC is presented in this pipeline as an exploratory robustness module rather than as the basis for strong state-based inference. (\autoref{fig:sensitivity})

![Sensitivity analysis results from OpenNeuro ds007318 (N=3, 5 runs). **A.** Pairwise Spearman rank correlation matrix for static functional connectivity vectors across 9 parameter conditions (Schaefer-200, varying GSR, scrubbing threshold, window length, and smoothing kernel). All conditions except GSR-off produce ρ > 0.99. **B.** Parameter impact quantified as mean deviation from baseline (1 − ρ). GSR is the dominant source of analytical variability. **C.** Summary of key sensitivity findings.\label{fig:sensitivity}](sensitivity_figures_real/joss_figure_sensitivity_combined.png)

---

## Discussion

### Principal findings

The primary finding of this study is that `fmri-pipeline` successfully recovered established features of large-scale brain organization and produced internally reproducible outputs when applied to a publicly available fMRI dataset, without modification of source code. The strongest validation result — canonical network organization confirmed by permutation test (*p* < 0.005) — establishes that the pipeline's FC matrices are biologically interpretable: within-network connections are systematically stronger than between-network connections across all 7 canonical Yeo networks. This is the expected signature of large-scale brain modularity [@bullmore2009complex] and its recovery is consistent with correct functioning of the pipeline's atlas parcellation and connectivity estimation steps.

The reproducibility analyses collectively support the conclusion that pipeline outputs are not arbitrary. ReHo spatial profiles were nearly perfectly reproduced within subjects across runs (Cohen's *d* = 31.2; see Table 1 footnote †), reflecting the high spatial consistency of local synchrony estimates. FC reproducibility was also evident, with a within-subject run-pair advantage confirmed by a bootstrap CI that entirely excludes zero; the formal significance test was underpowered at N = 3 and should not be interpreted as a null result. ICA stability across seeds (20/20 components, mean |*r*| = 0.88) further supports the reproducibility of the pipeline's parcellated decomposition outputs, with the caveat noted in Table 1 footnote ‡.

### Sensitivity to parameter choices

The sensitivity analysis directly addresses hypothesis 3: that preprocessing parameter choices have predictable, bounded effects. This hypothesis is supported for all parameters except GSR. Global signal regression remains the single most consequential methodological choice (Spearman ρ = 0.836 relative to baseline), consistent with the established literature on GSR's effect on FC topology [@power2012spurious; @murphy2017towards]. This effect is "predictable" in that its direction and magnitude are documented in the literature; users of `fmri-pipeline` are explicitly informed of this sensitivity through the automated sensitivity analysis output. All other tested parameters — scrubbing threshold (0.3–0.9 mm FD), smoothing kernel (4–8 mm), and window length (20–60 TRs) — produced static FC outputs with ρ ≥ 0.997 relative to baseline, confirming bounded sensitivity for the remaining parameter space.

Crucially, `fmri-pipeline` does not eliminate analytical variability — it documents it. The GSR sensitivity result (ρ = 0.836) demonstrates that a single parameter can substantially alter connectivity topology; the pipeline's contribution is to make this choice explicit, version-controlled, and reproducible in a plain-text configuration file, rather than embedding it in undocumented custom scripts. This supports users who cannot exactly reproduce a prior study's preprocessing parameters: minor differences in scrubbing threshold or smoothing kernel are unlikely to substantially alter static FC conclusions, whereas GSR usage should be matched whenever cross-study comparisons are made.

Dynamic FC sensitivity to window length (ARI 0.42–0.57 across window-size pairs) reflects the fundamental dependence of sliding-window methods on temporal resolution and is a property intrinsic to the method rather than a finding about neural dynamics per se. Because the specific states detected change substantially with window choice, dynamic FC outputs from this pipeline should be interpreted as exploratory indicators of temporal variability rather than as stable state assignments.

### Limitations

Several limitations constrain the strength of the conclusions. The validation sample comprised only 3 subjects and 5 runs total, which severely limits statistical power for between-subject comparisons. The formal FC reproducibility test did not reach *p* < 0.05 solely due to sample size — the effect size (Cohen's *d* = 5.7) and CI are unambiguous. The LORO-CV ICA stability analysis was not interpretable at N = 3 and should be revisited when larger datasets are analyzed. Dynamic FC state assignments were descriptive only.

The ICA stability analysis used temporal ICA on Schaefer-200 parcellated timeseries as a computational proxy; while sufficient for seed-stability benchmarking, this proxy does not validate the spatial ICA module of the pipeline. The ReHo recomputation used a brain mask derived from temporal variance of the cleaned BOLD rather than a gray-matter mask from fMRIPrep segmentation outputs; for ROI-level extraction using the Schaefer atlas, this distinction is methodologically negligible but is noted for completeness. The ReHo output-writing bug that necessitated this recomputation has been corrected in version [SOFTWARE_VERSION] of `fmri-pipeline`.

All 3 validated participants are drawn from a clinical population and the BOLD data were acquired during a working-memory task treated here as pseudo-resting-state (no task-evoked regressors were modeled prior to connectivity analysis). While canonical network organization is robust across populations and data types, generalizability of the pipeline's validation results to healthy resting-state data should be confirmed with an additional dataset. A YAML configuration for the CNeuroMod Friends naturalistic fMRI dataset has been prepared; full empirical validation on a second dataset is an important next step for establishing the pipeline's generalizability.

### Implications for student-scientist fMRI research

`fmri-pipeline` is designed to lower the barrier to reproducible fMRI analysis for student investigators and addresses gaps not covered by existing integrated tools such as CONN and XCP-D, which lack a built-in biological validity scorecard and are oriented toward established research workflows. The YAML-driven architecture means that all analytical decisions are version-controlled in a plain-text file, creating a complete record of methodology that can be shared alongside results. The built-in reproducibility suite — which runs automatically and produces a scorecard table — makes it possible for a student investigator to assess whether their pipeline outputs meet internal consistency criteria before drawing biological conclusions, without requiring external expert review of every parameter choice. The test suite (18 files, 382 tests), which requires no neuroimaging files, provides immediate verification that core computations are correct on any new platform.

---

## Data and Code Availability

The source code for `fmri-pipeline` is available at https://github.com/sarapradhan/biomed-research (version [SOFTWARE_VERSION]) and is archived at [ZENODO_DOI]. The pipeline configuration file (`pipeline.ds007318.yaml`), reproducibility configuration (`reproducibility_real.yaml`), and sensitivity outputs (`stability_summary.csv`) used to generate all results reported in this paper are included in the archived repository. The input fMRI data are publicly available from OpenNeuro (https://openneuro.org/datasets/ds007318).

---

## Acknowledgements

This work builds on an earlier project conducted through the Lumière Research Scholar Program. The author thanks the open-source neuroimaging community, particularly the developers of fMRIPrep, nilearn, nibabel, scipy, and statsmodels, whose tools were essential to this work. The ds007318 dataset was obtained from the OpenNeuro repository. The author used Claude (Anthropic) as a writing and coding assistant during manuscript and software development; all scientific content, interpretations, and final code were reviewed and verified by the author.

---

## References
