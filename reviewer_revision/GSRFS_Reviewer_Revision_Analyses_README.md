# GSR-FS reviewer-requested additional analyses

Manuscript: algorithms-4649834
Method: GSR-FS v0.16.2

This package contains the additional analyses run in response to reviewer comments 2, 3, 4, 5, 7, 8, 9 and 10.

Important methodological boundary
---------------------------------
The frozen primary GSR-FS definition was not changed. Sensitivity analyses vary
alpha, coverage_power, classifier, CGS threshold, or computational budgets only
to characterize behavior; they do not redefine the frozen primary estimator.

External benchmark reproduction
-------------------------------
The local reconstruction was first validated against the submitted 30-dataset
GSR-FS fixed-k results. Dataset-level means reproduce the submitted values.

Additional analyses
-------------------
1. Automatic-mode external validation:
   B=200, max_pairs=8000, same 30 datasets, same 3-fold splits and preprocessing.
   Empty selections were retained and evaluated with a prior-only classifier.

2. Component ablation:
   Full / No Coverage / No Residual / No Permutation, fixed-k, same 30-dataset panel.

3. Expanded automatic calibration:
   9 configurations: independent Gaussian, correlated Gaussian (rho=.5), and
   heavy-tailed t5; n,p = (100,20), (240,24), (500,50);
   50 replicates/configuration; B=100; max_pairs=5000.

4. Informative-vs-coherent-nuisance simulation:
   100 replicates/scenario; fixed-k and automatic modes.

5. Rare-geometry mitigation:
   alpha in {.05,.025,.01}; coverage_power in {2,3,4}; 50 replicates/configuration.

6. Classifier sensitivity:
   Logistic Regression, Linear SVM, Random Forest; same fixed and automatic subsets.

7. CGS threshold sensitivity:
   thresholds {.75,.80,.85,.90,.95}; 90 synthetic datasets plus 10 real datasets.

8. Scalability:
   runtime scaling over n, p, pair budget and permutation count; fresh-process
   peak RSS memory checks.

Key findings
------------
- Automatic vs submitted fixed-k:
  BA 0.6216 vs 0.6069; mean difference +0.0147; Wilcoxon p=0.1215.
  Automatic mean k=6.32; empty folds=7/90 (7.8%).

- Classifier sensitivity:
  Auto-minus-fixed mean BA was +0.0150 (Linear SVM), +0.0147 (Logistic),
  +0.0178 (Random Forest); none significant.

- External ablation:
  Removing residual increased geometry redundancy and reduced mean BA.
  After Holm correction, predictive differences were not significant, but geometry
  redundancy differences were significant.
  The no-permutation fixed-k variant had numerically higher predictive scores,
  showing that permutation calibration should be justified by null calibration and
  automatic stopping, not by fixed-k predictive superiority.

- Informative vs nuisance:
  Clean fixed-k scenario: precision 1.00.
  With coherent target-independent nuisance: nuisance selected in 79% of fixed-k
  runs and 100% of automatic runs. GSR-FS estimates structural rather than
  guaranteed target-specific relevance.

- Expanded calibration:
  Independent Gaussian pooled non-empty rate 0.0667; t5 0.0467.
  Correlated Gaussian pooled non-empty rate 0.9733, demonstrating that correlated
  predictor structure is treated as structural signal rather than an independence null.

- Rare-geometry mitigation:
  Original first 15 runs reproduced exactly: mean k=21.8, coverage=1.00,
  precision=0.0918.
  At alpha=.05, beta=4: mean k=10.44, precision=.2258, coverage=.95.
  At alpha=.01, beta=4: mean k=3.38, precision=.5275, coverage=.75.

- Residual targeted ablation:
  Full: mean k=2.00, redundancy=0.
  No residual: mean k=4.04, redundancy=.496.

- CGS threshold:
  Synthetic threshold .85: true-group recall=1.00, zero false assignments,
  mean stability gain=.549.
  On 10 real datasets at .85: mean exact Jaccard=.553, group-aware=.634,
  mean gain=.081; 7/10 datasets improved.

- Scalability:
  With pair budget fixed at 10k, n scaling is modest after the pair cap is reached.
  Runtime and memory increase strongly with p and pair budget.
  Fresh-process peak RSS at n=500,p=50 increased from ~165 MB (1k pairs)
  to ~259 MB (50k pairs).
