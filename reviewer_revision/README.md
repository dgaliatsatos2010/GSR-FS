# Reviewer-revision reproducibility artifacts

This directory contains machine-readable outputs and executable scripts generated
for the peer-review revision of the GSR-FS manuscript.

## Scope

The frozen primary GSR-FS algorithm remains version 0.16.2. The files in this
directory document additional validation and sensitivity analyses requested during
peer review; they do not constitute a new algorithm release.

## Contents

The directory includes artifacts for:

- expanded automatic-stopping calibration;
- structural-relevance versus target-relevance simulations;
- automatic-versus-fixed external validation;
- rare-geometry mitigation sensitivity;
- component ablation and Holm-adjusted statistical tests;
- downstream-classifier sensitivity;
- synthetic and real-data CGS threshold sensitivity;
- runtime and memory scalability analyses.

The revised Supplementary Material reports the corresponding Tables S20-S29 and
Figures S3-S8.

## Canonical software release

GSR-FS v0.16.2:
- GitHub repository: https://github.com/dgaliatsatos2010/GSR-FS
- Version DOI: 10.5281/zenodo.23046919
- Concept DOI: 10.5281/zenodo.23046348

## Reproducibility note

The reviewer-revision results were generated using the same frozen primary method.
Where analysis-specific values differ from the public library defaults (for example,
pair-budget or permutation-count reductions used for benchmarking), those settings
are explicitly reported in the revised manuscript and Supplementary Material.
