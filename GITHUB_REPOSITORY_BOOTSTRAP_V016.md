# GSR-FS v0.16 — GitHub repository bootstrap

This release is designed to be copied into a **dedicated new repository**, not into an unrelated project.

Recommended repository name: `GSR-FS` or `gsrfs`.

## Upload

Copy the contents of the GitHub-ready snapshot to the repository root. Do not upload `build/`, `dist/`, `*.egg-info/`, `.pytest_cache/`, or generated `benchmarks/results/v16/` outputs.

The workflow file must remain at:

```text
.github/workflows/cc18_v016.yml
```

## Actions

Open **Actions → GSR-FS v0.16 OpenML-CC18 evidence → Run workflow**.

Frozen defaults:

- accepted CC18 tasks: 30
- seeds: 0, 1, 2
- canonical set: Laplacian Score, SPEC, MCFS from pinned scikit-feature commit
- GSR residual power: 0.5
- pair sampler: balanced
- permutations: 50
- recent comparator: FGMRW-UFS TKDE 2026 on an X-defined tractable subset

An OpenML API key is optional for public downloads but can be added as repository secret `OPENML_API_KEY`.

## Integrity

The workflow clones scikit-feature outside the package tree at commit `48cffad4e88ff4b9d2f1c7baffb314d1b3303792`. FGMRW author source is downloaded only at runtime and verified against pinned Git blob SHA-1 values. Neither upstream source tree is redistributed by GSR-FS.

## Expected output

GitHub Actions uploads artifact `gsrfs-v016-cc18-canonical` containing `benchmarks/results/v16/`. The official CC18 and recent-author gates must be read from those generated outputs; the pre-run release does not claim they have passed.
