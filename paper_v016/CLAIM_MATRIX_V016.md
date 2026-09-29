# Claim matrix — v0.16

| Proposed statement | Status | Safe manuscript wording | Evidence / reason |
|---|---|---|---|
| GSR-FS is fully label-blind | **Supported** | “Feature selection uses only X; labels and ground-truth class count are excluded from selector configuration.” | Label-blindness audit and API/benchmark contracts. |
| Automatic GSR-FS is approximately calibrated under an independent null | **Supported with qualification** | “Empirical null calibration was consistent with the nominal 5% level.” | 13/200 = 6.5%; CI includes 5%; no theorem. |
| GSR-FS suppresses redundant representatives in controlled signal families | **Supported in simulation** | “Controlled redundant-signal simulations showed full group recovery with zero measured redundancy.” | 30/30 signal runs. |
| GSR-FS is competitive with classic canonical UFS methods | **Supported** | “Across 29 complete canonical real-data comparisons, performance was competitive with Laplacian Score, SPEC, and MCFS.” | Mean BA 0.6031; mean rank 2.862. |
| GSR-FS has the highest mean balanced accuracy in the complete canonical panel | **Supported descriptively** | “GSR-FS had the highest arithmetic mean BA, although SPEC had the best mean rank.” | 0.6031 vs 0.6019 SPEC. |
| GSR-FS significantly outperforms canonical methods | **Not supported** | Do not claim. | Friedman p=0.488; no Holm-significant win. |
| GSR-FS is state of the art | **Not supported** | Do not claim. | No significant superiority and recent-method benchmark pending. |
| GSR-FS replaces Laplacian Score/SPEC/MCFS | **Not supported** | Do not claim. | Replacement/superiority gate FAIL. |
| The 30-dataset panel is OpenML-CC18 | **False / prohibited** | “heterogeneous source-audited 30-dataset panel” | Formal CC18 workflow has not yet run. |
| GSR-FS has been empirically compared with FGMRW-UFS 2026 | **Not yet supported** | “A pinned FGMRW-UFS comparison is predeclared and source-audited.” | Recent-author empirical gate not executed. |
| CGS creates new latent features | **False** | “CGS is a diagnostic structural-substitution layer; transform returns actual selected columns.” | Structural-group API contract. |
