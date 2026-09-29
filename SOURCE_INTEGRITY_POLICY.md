# GSR-FS v0.13 Dataset Source-Integrity Policy

## Purpose

An unsupervised feature-selection benchmark is invalid if the feature matrix has already been
selected, reduced, sampled, or otherwise transformed using the evaluation target before the
unsupervised selector sees it. v0.13 therefore treats dataset provenance as part of the
publication evidence contract rather than as an informal data-loading detail.

## PASS

A source is eligible when:

- the original predictor information is retained (ordinary label-blind encoding is allowed);
- row subsampling, if any, is independent of `y`;
- feature filtering/selection before GSR-FS is independent of `y`;
- imputation, scaling, encoding, and dimensional handling are independent of `y`;
- the source URI/task/dataset identifier is recorded;
- provenance is explicit enough to audit.

## FAIL

A source is excluded when any pre-selector operation uses the target, including:

- Random Forest / classifier feature importance on `y`;
- supervised feature selection or dimensional reduction;
- class-stratified or response-histogram row reduction used to create the benchmark matrix;
- target-informed imputation, encoding, or scaling;
- any other operation that lets the evaluation target influence which predictor information
  reaches the unsupervised selector.

## REVIEW

A source with insufficient provenance is not counted toward the publication gate until the
processing history is resolved.

## Publication-gate rule

The `minimum_external_datasets >= 30` requirement counts only candidate-method datasets that
have an integrity audit status of `PASS`. A benchmark can therefore contain 30 result rows and
still fail the claim gate if even one of the required 30 sources is supervised-contaminated or
unresolved.
