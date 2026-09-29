# Label-blindness audit — v0.5

v0.5 corrects a methodological problem discovered in v0.4: the development MCFS comparator had
been configured using the number of ground-truth classes. Although the label vector itself was not
passed to MCFS, using `len(unique(y))` is still ground-truth leakage and is incompatible with the
strict no-label selection rule.

## Correction

- MCFS now defaults to `n_clusters="eigengap"`.
- Its spectral embedding dimension is estimated from the predictor graph only.
- The benchmark method factory no longer accepts a class count.
- SPEC, Laplacian Score, MaxVariance, MCFS and GSR-FS are all instantiated without `y`-derived
  hyperparameters.
- Ground-truth class count is used **only after feature selection** to evaluate KMeans partitions
  with NMI/ARI, which is a downstream external-evaluation convention rather than selector input.

## Empirical audit

On Iris, every bundled selector was fit repeatedly after independently permuting `y`. For all
methods and all audit trials, the selected feature indices were unchanged (`fraction_identical=1`).
The raw and summary CSV files are under `benchmarks/results/v05/`.

This empirical test is a regression guard, not a formal proof that arbitrary future code can never
access labels. The publication protocol therefore also requires code review of method factories.
