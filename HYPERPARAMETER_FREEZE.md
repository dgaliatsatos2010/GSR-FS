# GSR-FS v0.5 hyperparameter freeze

This file freezes the candidate defaults **before** the external OpenML-CC18 evaluation.
They may not be tuned independently on each final benchmark dataset.

## Frozen candidate defaults

- `n_features="auto"` for automatic-mode experiments; publication fixed-k experiments use the preregistered p-only grid.
- `n_permutations=200` for final automatic inference; lower values are allowed only for development/runtime screening and must be reported.
- `alpha=0.05`.
- `max_pairs=50000` for the library default; benchmark caps are reported explicitly when smaller.
- `robust=True`.
- `clip_z=4.5`.
- `coverage_power=2.0`.
- `use_residual=True`.
- **`residual_power=0.5`** (new v0.5 freeze).
- `allow_empty=True` in automatic mode.

## Why gamma = 0.5?

A development-only sensitivity study used controlled simulations plus the bundled Iris, Wine,
and Breast Cancer datasets. Gamma values `{0, 0.25, 0.5, 0.75, 1}` were compared before any
external CC18 run.

The full residual penalty (`gamma=1`) produced the strongest diversity pressure but materially
reduced conventional clustering NMI on the small real development set. `gamma=0.5` preserved
zero redundancy in the controlled signal groups while improving the dataset-balanced development
NMI from roughly 0.551 (`gamma=1`) to 0.649. It is therefore frozen as a compromise between
structural novelty and downstream clustering utility.

This choice is a development decision, not evidence that 0.5 is universally optimal.

## Final-evaluation rule

After a CC18 task is opened for final evaluation, no GSR-FS hyperparameter may be changed in
response to that task's labels or results. Any later change requires a new version and a fresh,
separately identified external benchmark.
