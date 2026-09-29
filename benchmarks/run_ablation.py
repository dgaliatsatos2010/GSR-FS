from pathlib import Path
import time
import numpy as np
import pandas as pd

from gsrfs import GSRSelector
from gsrfs.synthetic import SCENARIOS
from gsrfs.realdata import load_builtin_benchmarks, default_k_grid
from gsrfs.metrics import (
    group_coverage,
    relevant_precision,
    redundancy_rate,
    clustering_utility,
    knn_preservation,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

VARIANTS = {
    "GSR_full": dict(robust=True, coverage_power=2.0, use_residual=True),
    "GSR_no_SGC": dict(robust=True, coverage_power=0.0, use_residual=True),
    "GSR_no_residual": dict(robust=True, coverage_power=2.0, use_residual=False),
    "GSR_no_robust": dict(robust=False, coverage_power=2.0, use_residual=True),
}


def synthetic_ablation():
    rows = []
    for scenario, generator in SCENARIOS.items():
        if scenario == "null_independent":
            continue
        for seed in (0, 1, 2):
            X, y, groups, name = generator(seed=seed)
            k = len(groups)
            for variant, kwargs in VARIANTS.items():
                t0 = time.perf_counter()
                sel = GSRSelector(
                    n_features=k,
                    n_permutations=10,
                    max_pairs=6000,
                    random_state=seed,
                    **kwargs,
                ).fit(X)
                idx = sel.get_support(indices=True)
                u = clustering_utility(X, y, idx, len(np.unique(y)), random_state=seed)
                rows.append({
                    "scenario": name,
                    "seed": seed,
                    "variant": variant,
                    "group_coverage": group_coverage(idx, groups),
                    "relevant_precision": relevant_precision(idx, groups),
                    "redundancy_rate": redundancy_rate(idx, groups),
                    "ari": u["ari"],
                    "nmi": u["nmi"],
                    "fit_seconds": time.perf_counter() - t0,
                })
    return pd.DataFrame(rows)


def real_ablation():
    rows = []
    # Exclude Digits here to keep the ablation small; Digits remains in the main benchmark.
    for ds in load_builtin_benchmarks(include_digits=False):
        n_clusters = len(np.unique(ds.y))
        k_grid = default_k_grid(ds.X.shape[1], n_clusters)
        max_k = max(k_grid)
        for seed in (0, 1):
            for variant, kwargs in VARIANTS.items():
                t0 = time.perf_counter()
                sel = GSRSelector(
                    n_features=max_k,
                    n_permutations=10,
                    max_pairs=6000,
                    random_state=seed,
                    **kwargs,
                ).fit(ds.X)
                elapsed = time.perf_counter() - t0
                order = sel.get_support(indices=True)
                for k in k_grid:
                    idx = order[:k]
                    u = clustering_utility(ds.X, ds.y, idx, n_clusters, random_state=seed)
                    rows.append({
                        "dataset": ds.name,
                        "seed": seed,
                        "k": k,
                        "variant": variant,
                        "ari": u["ari"],
                        "nmi": u["nmi"],
                        "silhouette": u["silhouette"],
                        "knn_preservation": knn_preservation(ds.X, idx),
                        "fit_seconds": elapsed,
                    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    syn = synthetic_ablation()
    real = real_ablation()
    syn.to_csv(OUT / "ablation_synthetic_raw.csv", index=False)
    real.to_csv(OUT / "ablation_real_raw.csv", index=False)

    syn_summary = syn.groupby("variant")[[
        "group_coverage", "relevant_precision", "redundancy_rate", "ari", "nmi", "fit_seconds"
    ]].mean().sort_values("group_coverage", ascending=False)
    real_summary = real.groupby("variant")[[
        "ari", "nmi", "silhouette", "knn_preservation", "fit_seconds"
    ]].mean().sort_values("nmi", ascending=False)
    syn_summary.to_csv(OUT / "ablation_synthetic_summary.csv")
    real_summary.to_csv(OUT / "ablation_real_summary.csv")
    print("SYNTHETIC ABLATION")
    print(syn_summary)
    print("\nREAL ABLATION")
    print(real_summary)
