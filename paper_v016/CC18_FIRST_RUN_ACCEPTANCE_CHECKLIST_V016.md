# First official CC18 run — acceptance checklist v0.16

The official standardized-suite result is accepted only if **all** items below are satisfied.

- [ ] Workflow file is `.github/workflows/cc18_v016.yml`.
- [ ] Repository commit/tag for the run is recorded.
- [ ] GSR-FS version is 0.16.0 and the frozen score/hyperparameters are unchanged.
- [ ] Frozen CC18 suite/task manifest is used; task acceptance is independent of y and downstream performance.
- [ ] At least the predeclared number of source-integrity-eligible CC18 tasks completes.
- [ ] Every accepted task has a source-integrity PASS record.
- [ ] Labels are used only for downstream evaluation, never selector configuration.
- [ ] scikit-feature is checked out at the pinned commit recorded in the source lock.
- [ ] Canonical Laplacian Score, SPEC, and MCFS provenance is preserved.
- [ ] MCFS does not receive the ground-truth number of classes.
- [ ] Raw rows, task manifest, exclusions, failures, runtimes, environment metadata, seeds, and dataset-wise summaries are archived.
- [ ] Dataset-balanced ranks are computed before omnibus/post-hoc tests.
- [ ] Friedman, paired Wilcoxon, Holm adjustment, and rank-biserial effects are reported.
- [ ] No hyperparameter is changed after inspecting CC18 labels/results.
- [ ] FGMRW-UFS source commit/blob identifiers match the frozen recent-author lock.
- [ ] FGMRW-UFS receives training X only; its tractable task subset is chosen by the frozen X-only rule.
- [ ] The final report distinguishes standardized-suite evidence from the earlier heterogeneous panel.
- [ ] Strong-superiority language is used only if the predeclared superiority gate actually passes.
