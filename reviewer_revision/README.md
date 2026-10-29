# Reviewer-revision reproducibility artifacts

This directory contains machine-readable outputs and executable scripts generated
for the peer-review revision of the GSR-FS manuscript.

## Scope

The frozen primary GSR-FS algorithm remains version 0.16.2. The files in this
directory document additional validation and sensitivity analyses requested during
peer review; they do not constitute a new algorithm release or modify the frozen
mathematical objective.

## Contents

The directory includes artifacts for:

- expanded automatic-stopping calibration;
- structural-relevance versus target-relevance simulations;
- automatic-versus-fixed external validation;
- rare-geometry mitigation sensitivity;
- component ablation and Holm-adjusted statistical tests;
- downstream-classifier sensitivity;
- synthetic and real-data CGS threshold sensitivity;
- runtime and memory scalability analyses;
- Reviewer-2 high-precision independent-null calibration;
- Reviewer-2 permutation-count sensitivity;
- Reviewer-2 robust clipping-threshold sensitivity;
- Reviewer-2 comparative structural-recovery experiments;
- Reviewer-2 high-dimensional scalability through 10,000 candidate features.

Reviewer-2-specific files use the `r2_` or `run_r2_` prefix. The comparative
structural-recovery experiments use fixed cardinality for all methods so that
recovery and redundancy are compared without confounding by different stopping
rules.

## Reviewer-2 validation design

The high-precision independent-Gaussian calibration used n=240, p=24 and 500
independent datasets with B=200 permutations and a 5,000-pair budget. The same
runs were summarized at alpha values 0.01, 0.025, 0.05 and 0.10.

Permutation-count sensitivity used common seeds and B in {100, 200, 500, 1000}.
Clipping sensitivity used clip_z in {3.0, 4.0, 4.5, 5.0, 6.0} under coherent
rare-geometry conditions.

Comparative structural recovery used n=300 and fixed k=5. The moderate regime
used p=100 with five latent groups, four correlated proxies per group and 80
noise variables; the larger regime used p=500 with five latent groups, ten
correlated proxies per group and 450 noise variables. Comparators were
MaxVariance, Laplacian Score, SPEC and label-blind eigengap MCFS.

The high-dimensional engineering stress test used n=250, fixed k=8,
B=25 and max_pairs=2000 for p in {500, 1000, 2500, 5000, 10000}. These capped
computational settings characterize feasibility and are not presented as a
replacement for the frozen default calibration.

## Canonical software release

GSR-FS v0.16.2:
- GitHub repository: https://github.com/dgaliatsatos2010/GSR-FS
- Version DOI: 10.5281/zenodo.23046919
- Concept DOI: 10.5281/zenodo.23046348

## Reproducibility note

The reviewer-revision results were generated using the same frozen primary
method. Where analysis-specific values differ from the public library defaults
(for example, pair-budget or permutation-count reductions used for benchmarking),
those settings are explicitly recorded in the machine-readable outputs and
analysis scripts. `MANIFEST_SHA256.csv` provides file sizes and SHA-256 hashes
for the complete reviewer-revision snapshot.
