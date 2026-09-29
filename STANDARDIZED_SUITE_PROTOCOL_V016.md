# Standardized-suite protocol — v0.16

The formal suite remains **OpenML-CC18, suite ID 99**, using the frozen 72-task manifest already carried by GSR-FS.

The GitHub Actions workflow `.github/workflows/cc18_v016.yml` performs two independent runs on a normal networked runner:

1. **Formal canonical CC18 run:** first 30 source-integrity-eligible tasks in the frozen suite order, frozen GSR-FS plus pinned scikit-feature Laplacian Score, SPEC, and MCFS.
2. **Recent-author run:** first 10 frozen CC18 tasks satisfying the X-only tractability rule `encoded p <= 100` and uniform `n <= 1500`, comparing frozen GSR-FS with pinned FGMRW-UFS TKDE 2026.

Task acceptance never uses labels, class counts, downstream metrics, or method success. Labels are evaluation-only. The v0.14 heterogeneous 30-dataset panel remains separate and must not be described as CC18.
