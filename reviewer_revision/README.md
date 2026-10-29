# Reviewer-Revision Reproducibility Artifacts

This directory contains machine-readable outputs and executable Python scripts generated for the peer-review revision of the GSR-FS manuscript:

**Geometry-Supported Residual Feature Selection: A Label-Blind Algorithm for Structural and Redundancy-Aware Feature Selection**

## Scope

The frozen primary GSR-FS algorithm remains version **0.16.2**. The files in this directory document additional validation and sensitivity analyses requested during peer review and do not constitute a new algorithm release.

## Included analyses

The directory contains artifacts for:

- expanded automatic-stopping calibration;
- structural-relevance versus target-relevance simulations;
- automatic-versus-fixed external validation;
- rare-geometry mitigation sensitivity;
- component ablation and Holm-adjusted statistical tests;
- downstream-classifier sensitivity;
- synthetic and real-data CGS threshold sensitivity;
- runtime and memory scalability analyses.

The revised Supplementary Material reports these analyses as **Tables S20–S29** and **Figures S3–S8**.

## Canonical software release

GSR-FS v0.16.2:

- GitHub repository: https://github.com/dgaliatsatos2010/GSR-FS
- Version DOI: 10.5281/zenodo.23046919
- Concept DOI: 10.5281/zenodo.23046348

## Reproducibility note

The reviewer-revision analyses were generated using the same frozen primary GSR-FS method. Analysis-specific values differing from the public library defaults, including benchmark-specific permutation counts or pair budgets, are explicitly reported in the revised manuscript and Supplementary Material.
