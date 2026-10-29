# GSR-FS reviewer-requested additional analyses

Manuscript: algorithms-4649834  
Method: GSR-FS v0.16.2

This package contains additional analyses conducted during peer review. The
original Reviewer-1 analyses are retained unchanged, and the Reviewer-2 analyses
are added as a separate, traceable extension.

## Important methodological boundary

The frozen primary GSR-FS definition was not changed. Sensitivity analyses vary
alpha, coverage_power, classifier, CGS threshold, robust clipping, or
computational budgets only to characterize behavior; they do not redefine the
frozen primary estimator.

## External benchmark reproduction

The local reconstruction was first validated against the submitted 30-dataset
GSR-FS fixed-k results. Dataset-level means reproduce the submitted values.

## Reviewer-1 additional analyses

1. Automatic-mode external validation  
   B=200, max_pairs=8000, same 30 datasets, same 3-fold splits and preprocessing.
   Empty selections were retained and evaluated with a prior-only classifier.

2. Component ablation  
   Full / No Coverage / No Residual / No Permutation, fixed-k, same 30-dataset panel.

3. Expanded automatic calibration  
   Independent Gaussian, correlated Gaussian (rho=.5), and heavy-tailed t5;
   (n,p)=(100,20), (240,24), (500,50); 50 replicates/configuration;
   B=100; max_pairs=5000.

4. Informative-vs-coherent-nuisance simulation  
   100 replicates/scenario; fixed-k and automatic modes.

5. Rare-geometry mitigation  
   alpha in {.05,.025,.01}; coverage_power in {2,3,4}; 50 replicates/configuration.

6. Classifier sensitivity  
   Logistic Regression, Linear SVM, Random Forest; same fixed and automatic subsets.

7. CGS threshold sensitivity  
   thresholds {.75,.80,.85,.90,.95}; 90 synthetic datasets plus 10 real datasets.

8. Scalability  
   Runtime scaling over n, p, pair budget and permutation count; fresh-process
   peak RSS memory checks.

## Reviewer-2 additional analyses

9. High-precision independent-null calibration  
   n=240, p=24, 500 independent Gaussian datasets, B=200,
   max_pairs=5000, coverage_power=2, residual_power=0.5.
   Results were summarized for alpha in {0.01, 0.025, 0.05, 0.10}.

10. Permutation-count sensitivity  
    Common-seed null experiments with B in {100, 200, 500, 1000};
    50 replicates per B. Monte Carlo p-value resolution and threshold behavior
    are reported explicitly.

11. Robust clipping sensitivity  
    clip_z in {3.0, 4.0, 4.5, 5.0, 6.0}; 30 replicates/configuration under
    coherent rare-geometry conditions. This analysis characterizes the frozen
    c=4.5 engineering default rather than claiming theoretical optimality.

12. Comparative structural-recovery benchmark  
    All methods used fixed k=5 and n=300.
    - moderate_p100: p=100, five latent groups, four correlated proxies/group,
      80 noise variables, 60 independent replicates;
    - larger_p500: p=500, five latent groups, ten correlated proxies/group,
      450 noise variables, 40 independent replicates.
    GSR-FS used B=200; max_pairs=5000 for p=100 and 2000 for p=500.
    Comparators were MaxVariance, Laplacian Score, SPEC, and label-blind
    eigengap MCFS.

13. High-dimensional engineering scalability  
    n=250, fixed k=8, B=25, max_pairs=2000, three fresh-process runs for
    p in {500, 1000, 2500, 5000, 10000, 20000}. Wall-clock runtime, peak resident-set
    memory and completion status are reported.

## Key Reviewer-1 findings

- Automatic vs submitted fixed-k:
  BA 0.6216 vs 0.6069; mean difference +0.0147; Wilcoxon p=0.1215.
  Automatic mean k=6.32; empty folds=7/90 (7.8%).

- Classifier sensitivity:
  Auto-minus-fixed mean BA was +0.0150 (Linear SVM), +0.0147 (Logistic),
  +0.0178 (Random Forest); none significant.

- External ablation:
  Removing residual increased geometry redundancy and reduced mean BA.
  After Holm correction, predictive differences were not significant, but
  geometry-redundancy differences were significant.

- Informative vs nuisance:
  Clean fixed-k scenario: precision 1.00.
  With coherent target-independent nuisance: nuisance selected in 79% of
  fixed-k runs and 100% of automatic runs.

- Expanded calibration:
  Independent Gaussian pooled non-empty rate 0.0667; t5 0.0467.
  Correlated Gaussian pooled non-empty rate 0.9733.

- Rare-geometry mitigation:
  At alpha=.05, beta=4: mean k=10.44, precision=.2258, coverage=.95.
  At alpha=.01, beta=4: mean k=3.38, precision=.5275, coverage=.75.

- Residual targeted ablation:
  Full: mean k=2.00, redundancy=0.
  No residual: mean k=4.04, redundancy=.496.

- CGS threshold:
  Synthetic threshold .85: true-group recall=1.00, zero false assignments.
  On 10 real datasets at .85: mean exact Jaccard=.553,
  group-aware=.634, mean gain=.081; 7/10 datasets improved.

## Key Reviewer-2 findings

- High-precision null calibration:
  at alpha=.05, 25/500 datasets produced non-empty selections,
  empirical rate=.050 with exact 95% CI .0326-.0729.
  The rates at alpha=.01, .025, .05 and .10 were .010, .026, .050 and .100.

- Permutation sensitivity:
  Monte Carlo resolution improved from 1/101=.00990 at B=100 to
  1/1001=.00100 at B=1000. Non-empty rates were .04, .04, .02 and .02
  for B=100, 200, 500 and 1000 in the 50 common-seed runs.

- Clipping sensitivity:
  clip_z values 3-5 produced broadly similar rare-geometry behavior.
  The frozen c=4.5 setting is therefore treated as a reproducible engineering
  default rather than a universal optimum.

- Comparative structural recovery, p=100:
  GSR-FS mean group coverage=.8533, relevant precision=1.0000 and
  redundancy=.1467. MCFS coverage=.5867 and Laplacian Score coverage=.5433.

- Comparative structural recovery, p=500:
  GSR-FS mean group coverage=.7600 and redundancy=.0000.
  MCFS mean coverage=.5950 with redundancy=.4000;
  Laplacian Score mean coverage=.3500 with redundancy=.6500.
  These findings support a redundancy-control advantage rather than a claim of
  universal superiority across every metric.

- High-dimensional scalability:
  all p configurations through p=10000 completed successfully under capped
  B=25 and max_pairs=2000. Median runtime increased from .652 s at p=500
  to 14.835 s at p=10000; median peak RSS increased from 218.7 MB to
  1003.4 MB.

## File naming

Reviewer-2 outputs use the `r2_` prefix. Reviewer-2 executable scripts use the
`run_r2_` prefix. `run_r2_highdim_scalability.py` orchestrates the complete 15-run high-dimensional benchmark and writes the published raw and summary CSVs. `r2_baselines.py` contains the comparator implementations used
by the controlled structural-recovery analysis.

`MANIFEST_SHA256.csv` records file size and SHA-256 hash for every file in the
reviewer-revision snapshot except the manifest itself.


## FGMRW-UFS comparator extension

A recent 2026 unsupervised feature-selection comparator, FGMRW-UFS
(IEEE Transactions on Knowledge and Data Engineering,
DOI 10.1109/TKDE.2026.3700270), was evaluated under the same fixed-k
structural-recovery scenarios. The public repository state was pinned at
commit 325a904a28284a879c8a522ca7c4abc56449e9f5. The author test settings
alpha=0.1 and s=0.8 were retained, no target labels were supplied to the
selector, and all methods selected k=5 features. Full raw and summary outputs
are provided for the 60 p=100 and 40 p=500 replicates.

The FGMRW-UFS results are included as a contemporary comparator, not as evidence
that clustering-oriented UFS methods should optimize the same structural-group
recovery criterion as GSR-FS.
