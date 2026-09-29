# v0.11 CGS diagnostic freeze

## Purpose

v0.11 does **not** alter the frozen GSR-FS selection objective. It introduces a post-selection, fully label-blind diagnostic called **Contextual Geometry Substitution (CGS)** to determine whether an unselected feature can act as a geometry-level substitute for a selected representative.

For a frozen selected set `S`, selected representative `r in S`, candidate `j not in S`, and unit-norm pair-geometry profiles `h`, define the non-negative residual fraction `R(a | B)` as the squared reconstruction residual of profile `a` from basis `B`. Then

\[
CGS(r,j\mid S)
=1-\max\left\{
R(j\mid S),
R(r\mid S\setminus\{r\}\cup\{j\})
\right\}.
\]

A value near one means that (i) the candidate is already representable by the selected geometry basis and (ii) replacing `r` by `j` still reconstructs the representative's geometry.

## Threshold development

The grouping threshold was developed using only synthetic and null data. Candidate thresholds were `0.80, 0.85, 0.90, 0.95, 0.98`.

Across 15 synthetic datasets (redundant blocks, nonlinear moons, three-factor blobs; five seeds each), exact-column resampling Jaccard averaged **0.400**. At threshold 0.85:

- substitution-aware Jaccard: **0.987**;
- absolute stability gain: **+0.587**;
- true redundant-group recovery: **1.000**;
- noise assignment rate: **0.000**.

Threshold 0.80 gave the same synthetic summary, but 0.85 was frozen as the more conservative value.

## Null diagnostic

Across 50 independent-null datasets (`n=160`, `p=20`, fixed `k=3`), threshold 0.85 produced:

- mean unselected-feature assignment rate: **0.000**;
- datasets with any substitution assignment: **0/50**;
- mean maximum unselected CGS similarity: approximately **0.253**;
- 95th percentile of maximum unselected similarity: approximately **0.276**.

These empirical results do not constitute a formal false-grouping guarantee.

## Freeze decision

The experimental v0.11 default diagnostic threshold is therefore frozen at:

\[
\boxed{CGS\ threshold=0.85}
\]

before examining the fresh real structural panel.

CGS remains diagnostic: it does not alter `GSRSelector` scores, ranking, or selected columns.
