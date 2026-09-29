# Reproducibility statement — v0.16

The GSR-FS research package records frozen method settings, source-integrity audits, canonical source locks, benchmark manifests, raw result tables, dataset-balanced summaries, post-hoc statistics, and automated claim gates. The core selector uses independent random streams for pair sampling and permutation calibration, a shared permutation bank across features, and deterministic nested balanced-pair prefixes for a fixed seed.

The heterogeneous 30-dataset evidence panel and the 29-dataset complete canonical comparison are already archived in the research release. Canonical Laplacian Score, SPEC, and MCFS were executed from independently sourced scikit-feature code; third-party source is not redistributed in the MIT package.

The next standardized confirmation is predeclared in `.github/workflows/cc18_v016.yml`. The workflow pins the scikit-feature repository commit, runs the frozen OpenML-CC18 protocol, records source-integrity for accepted tasks, and executes a separate source-audited FGMRW-UFS 2026 comparison on an X-defined tractable subset. At the v0.16 freeze, those networked runs have not yet been executed and must not be described as completed evidence.
