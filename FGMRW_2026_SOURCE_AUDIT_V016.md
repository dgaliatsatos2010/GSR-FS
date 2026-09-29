# FGMRW-UFS 2026 source audit — v0.16

GSR-FS v0.16 adds a **non-vendoring bridge** to the public repository `HongtaoGao-code/FGMRW-UFS`, pinned to commit `325a904a28284a879c8a522ca7c4abc56449e9f5`.

## Upstream files locked

- `FGMRW-UFS-code.py` Git blob SHA-1: `ff817880c7aa56366e957fe33e751b0db273443d`
- `GB.py` Git blob SHA-1: `dd4814cdae5395ccaf35eb2c697e06efdbfb8bfb`

No upstream source is included in the GSR-FS wheel or release ZIP. The repository root viewed during the audit exposed `Example.mat`, `FGMRW-UFS-code.py`, `GB.py`, and `README.md`, but no explicit LICENSE file. Therefore source redistribution is intentionally avoided.

## Label-blind execution boundary

The public `FGMRW_UFS` function documents `data` as a matrix **without labels**. The public demo, however, calls a helper that retains the final column and then passes the resulting matrix to `FGMRW_UFS`. v0.16 does **not** reproduce that ambiguous demo data path. The bridge passes training-fold `X` only; `y` and ground-truth class count are never passed.

The public implementation treats a column as numerical under its Gaussian-kernel branch when its minimum is exactly 0 and maximum exactly 1. v0.16 therefore fits MinMax scaling on training-fold `X` only before invoking the pinned author function. The example parameters `alpha=0.1` and `s=0.8` are frozen without downstream-label tuning.

## Claim boundary

Until the networked workflow actually completes, FGMRW is **source-audited and execution-ready**, not counted as an executed recent-author comparator in the current empirical superiority claim.
