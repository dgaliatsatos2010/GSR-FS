from __future__ import annotations

import time
import numpy as np
import pandas as pd

from .selector import GSRSelector
from .baselines import (
    MaxVarianceSelector, LaplacianScoreSelector, SPECSelector, MCFSSelector
)
from .metrics import group_coverage, redundancy_rate, relevant_precision, clustering_utility, knn_preservation, geometry_redundancy, normalized_effective_rank
from .synthetic import SCENARIOS


def _implementation_kind(source):
    text = str(source or "").lower()
    if text.startswith("author_code:") or "author" in text:
        return "author_code"
    if "scikit-feature" in text or text.startswith("canonical:"):
        return "canonical"
    return "native_or_local"


def _methods(k, seed, gsr_permutations=20, max_pairs=12000):
    """Construct strictly label-blind selectors for a fixed-k comparison.

    No method receives y or the ground-truth class count.  MCFS estimates its
    spectral embedding dimension from X internally via an eigengap rule.
    """
    return {
        "GSR-FS": GSRSelector(
            n_features=k,
            n_permutations=gsr_permutations,
            max_pairs=max_pairs,
            pair_sampler="balanced",
            random_state=seed,
            support_engine="vectorized",
            residual_engine="auto",
        ),
        "MaxVariance": MaxVarianceSelector(n_features=k),
        "LaplacianScore": LaplacianScoreSelector(n_features=k, n_neighbors=7),
        "SPEC": SPECSelector(n_features=k, gamma=1.0, style=0, standardize=False),
        "MCFS": MCFSSelector(
            n_features=k,
            n_clusters="eigengap",
            n_neighbors=7,
            alpha=0.003,
            max_clusters=10,
            random_state=seed,
        ),
    }


def run_synthetic_benchmark(
    scenarios=None,
    seeds=(0, 1, 2),
    gsr_permutations=20,
    max_pairs=12000,
):
    scenarios = list(SCENARIOS) if scenarios is None else list(scenarios)
    rows = []
    for scenario in scenarios:
        generator = SCENARIOS[scenario]
        for seed in seeds:
            X, y, groups, scenario_name = generator(seed=seed)
            n_clusters = len(np.unique(y))
            k = len(groups) if groups else min(2, X.shape[1])
            for method_name, selector in _methods(k, seed, gsr_permutations, max_pairs).items():
                t0 = time.perf_counter()
                try:
                    selector.fit(X)
                    elapsed = time.perf_counter() - t0
                    selected = selector.get_support(indices=True)
                    util = clustering_utility(X, y, selected, n_clusters, random_state=seed)
                    rows.append({
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method_name,
                        "n_samples": X.shape[0],
                        "n_features": X.shape[1],
                        "k": k,
                        "selected": ",".join(map(str, selected.tolist())),
                        "group_coverage": group_coverage(selected, groups),
                        "relevant_precision": relevant_precision(selected, groups),
                        "redundancy_rate": redundancy_rate(selected, groups),
                        "ari": util["ari"],
                        "nmi": util["nmi"],
                        "silhouette": util["silhouette"],
                        "fit_seconds": elapsed,
                        "error": "",
                    })
                except Exception as exc:
                    rows.append({
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": method_name,
                        "n_samples": X.shape[0],
                        "n_features": X.shape[1],
                        "k": k,
                        "selected": "",
                        "group_coverage": np.nan,
                        "relevant_precision": np.nan,
                        "redundancy_rate": np.nan,
                        "ari": np.nan,
                        "nmi": np.nan,
                        "silhouette": np.nan,
                        "fit_seconds": time.perf_counter() - t0,
                        "error": repr(exc),
                    })
    return pd.DataFrame(rows)


def summarize(df):
    metrics = ["group_coverage", "relevant_precision", "redundancy_rate", "ari", "nmi", "silhouette", "fit_seconds"]
    return (
        df.groupby("method", dropna=False)[metrics]
        .agg(["mean", "std"])
        .sort_values(("group_coverage", "mean"), ascending=False)
    )




def run_real_benchmark(
    datasets=None,
    seeds=(0, 1, 2),
    gsr_permutations=20,
    max_pairs=12000,
):
    """Run a reproducible fixed-k unsupervised benchmark on real datasets.

    Feature selectors are fit on X only. Ground-truth labels are used afterwards
    solely to score KMeans partitions with ARI/NMI. The kNN preservation metric is
    entirely label-free.

    For efficiency, each method is fit once per dataset/seed at the largest k in
    the preregistered grid, then smaller k values use prefixes of that fitted
    ranking/greedy path. This is exact for the bundled selectors because their
    ranking/greedy order does not depend on the requested truncation length.
    """
    from .realdata import load_builtin_benchmarks, publication_k_grid

    if datasets is None:
        datasets = load_builtin_benchmarks(include_digits=True)

    rows = []
    for ds in datasets:
        X, y = ds.X, ds.y
        n_clusters = len(np.unique(y))
        k_grid = publication_k_grid(X.shape[1])
        max_k = max(k_grid)

        for seed in seeds:
            methods = _methods(max_k, seed, gsr_permutations, max_pairs)
            fitted = {}
            for method_name, selector in methods.items():
                t0 = time.perf_counter()
                try:
                    selector.fit(X)
                    elapsed = time.perf_counter() - t0
                    order = selector.get_support(indices=True).tolist()
                    fitted[method_name] = (order, elapsed, "")
                except Exception as exc:
                    fitted[method_name] = ([], time.perf_counter() - t0, repr(exc))

            for k in k_grid:
                for method_name in methods:
                    order, elapsed, err = fitted[method_name]
                    if err:
                        rows.append({
                            "dataset": ds.name,
                            "panel_status": getattr(ds, "panel_status", "unknown"),
                            "source": getattr(ds, "source", ""),
                            "seed": seed,
                            "method": method_name,
                            "n_samples": X.shape[0],
                            "n_features": X.shape[1],
                            "n_classes": n_clusters,
                            "k": k,
                            "selected": "",
                            "ari": np.nan,
                            "nmi": np.nan,
                            "silhouette": np.nan,
                            "knn_preservation": np.nan,
                            "geometry_redundancy": np.nan,
                            "normalized_effective_rank": np.nan,
                            "fit_seconds": elapsed,
                            "error": err,
                        })
                        continue

                    selected = np.asarray(order[:k], dtype=int)
                    util = clustering_utility(
                        X, y, selected, n_clusters, random_state=seed
                    )
                    rows.append({
                        "dataset": ds.name,
                        "panel_status": getattr(ds, "panel_status", "unknown"),
                        "source": getattr(ds, "source", ""),
                        "seed": seed,
                        "method": method_name,
                        "n_samples": X.shape[0],
                        "n_features": X.shape[1],
                        "n_classes": n_clusters,
                        "k": k,
                        "selected": ",".join(map(str, selected.tolist())),
                        "ari": util["ari"],
                        "nmi": util["nmi"],
                        "silhouette": util["silhouette"],
                        "knn_preservation": knn_preservation(X, selected, n_neighbors=10),
                        "geometry_redundancy": geometry_redundancy(X, selected, random_state=seed),
                        "normalized_effective_rank": normalized_effective_rank(X, selected),
                        # Report one fit cost for every k because the ranking can be
                        # reused. Consumers should not sum this column across k.
                        "fit_seconds": elapsed,
                        "error": "",
                    })
    return pd.DataFrame(rows)

def summarize_real(df):
    """Dataset-balanced summary of real-data benchmark metrics.

    Repeated k/seed rows are first averaged within each dataset/method, then
    datasets receive equal weight. This avoids overweighting datasets that happen
    to contribute more k values or repeated runs.
    """
    metrics = ["ari", "nmi", "silhouette", "knn_preservation", "geometry_redundancy", "normalized_effective_rank", "fit_seconds"]
    per_dataset = (
        df.groupby(["dataset", "method"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    return (
        per_dataset.groupby("method", dropna=False)[metrics]
        .agg(["mean", "std"])
        .sort_values(("nmi", "mean"), ascending=False)
    )


def publication_method_factories(kind="development"):
    """Return method factories for publication-style real-data benchmarking.

    Parameters
    ----------
    kind : {"development", "canonical"}
        ``development`` uses bundled clean-room comparators. ``canonical`` uses
        optional bridges to an independently installed scikit-feature package.
        In both cases every factory accepts ``(k, seed)`` and no factory receives
        y or a ground-truth class count.
    """
    if kind == "development":
        return {
            "GSR-FS": lambda k, seed: GSRSelector(
                n_features=k, n_permutations=50, max_pairs=50000,
                pair_sampler="balanced", random_state=seed,
                support_engine="vectorized", residual_engine="auto",
            ),
            "MaxVariance": lambda k, seed: MaxVarianceSelector(n_features=k),
            "LaplacianScore-dev": lambda k, seed: LaplacianScoreSelector(
                n_features=k, n_neighbors=7
            ),
            "SPEC-dev": lambda k, seed: SPECSelector(n_features=k),
            "MCFS-eigengap-dev": lambda k, seed: MCFSSelector(
                n_features=k, n_clusters="eigengap", n_neighbors=7,
                alpha=0.003, max_clusters=10, random_state=seed,
            ),
        }
    if kind == "canonical":
        from .canonical import (
            ScikitFeatureLaplacianSelector,
            ScikitFeatureSPECSelector,
            ScikitFeatureMCFSSelector,
        )
        return {
            "GSR-FS": lambda k, seed: GSRSelector(
                n_features=k, n_permutations=50, max_pairs=50000,
                pair_sampler="balanced", random_state=seed,
                support_engine="vectorized", residual_engine="auto",
            ),
            "MaxVariance": lambda k, seed: MaxVarianceSelector(n_features=k),
            "LaplacianScore-scikit-feature": lambda k, seed: ScikitFeatureLaplacianSelector(n_features=k),
            "SPEC-scikit-feature": lambda k, seed: ScikitFeatureSPECSelector(n_features=k, style=0),
            # Fixed 5 is the external implementation's default and is label-blind.
            "MCFS-scikit-feature-k5": lambda k, seed: ScikitFeatureMCFSSelector(n_features=k, n_clusters=5),
        }
    raise ValueError("kind must be 'development' or 'canonical'.")


def run_real_benchmark_flexible(
    datasets,
    method_factories,
    seeds=(0, 1, 2),
):
    """Run real-data benchmarking with method-specific ranking semantics.

    Methods whose estimator exposes ``ranking_depends_on_k=True`` are re-fit for
    every fixed-k value. Other methods are fit once at the largest k and their
    ranking prefix is reused. This distinction is required for canonical MCFS,
    whose LARS cardinality constraint is itself the requested feature count.
    """
    from .realdata import publication_k_grid

    rows = []
    for ds in datasets:
        X, y = ds.X, ds.y
        n_clusters = len(np.unique(y))
        k_grid = publication_k_grid(X.shape[1])
        max_k = max(k_grid)

        for seed in seeds:
            for method_name, factory in method_factories.items():
                probe = factory(max_k, seed)
                k_dependent = bool(getattr(probe, "ranking_depends_on_k", False))
                cached = None
                if not k_dependent:
                    t0 = time.perf_counter()
                    try:
                        probe.fit(X)
                        cached = (
                            probe.get_support(indices=True).tolist(),
                            time.perf_counter() - t0,
                            "",
                            getattr(probe, "external_source_", "native_or_local"),
                        )
                    except Exception as exc:
                        cached = ([], time.perf_counter() - t0, repr(exc), "")

                for k in k_grid:
                    if k_dependent:
                        selector = factory(k, seed)
                        t0 = time.perf_counter()
                        try:
                            selector.fit(X)
                            order = selector.get_support(indices=True).tolist()
                            elapsed = time.perf_counter() - t0
                            err = ""
                            source = getattr(selector, "external_source_", "external")
                        except Exception as exc:
                            order, elapsed, err, source = [], time.perf_counter() - t0, repr(exc), ""
                    else:
                        order, elapsed, err, source = cached

                    if err:
                        rows.append({
                            "dataset": ds.name, "seed": seed, "method": method_name,
                            "n_samples": X.shape[0], "n_features": X.shape[1],
                            "n_classes": n_clusters, "k": k, "selected": "",
                            "ari": np.nan, "nmi": np.nan, "silhouette": np.nan,
                            "knn_preservation": np.nan, "geometry_redundancy": np.nan,
                            "normalized_effective_rank": np.nan, "fit_seconds": elapsed,
                            "ranking_depends_on_k": k_dependent, "implementation_source": source,
                            "implementation_kind": _implementation_kind(source),
                            "error": err,
                        })
                        continue

                    selected = np.asarray(order[:k], dtype=int)
                    util = clustering_utility(X, y, selected, n_clusters, random_state=seed)
                    rows.append({
                        "dataset": ds.name, "seed": seed, "method": method_name,
                        "n_samples": X.shape[0], "n_features": X.shape[1],
                        "n_classes": n_clusters, "k": k,
                        "selected": ",".join(map(str, selected.tolist())),
                        "ari": util["ari"], "nmi": util["nmi"],
                        "silhouette": util["silhouette"],
                        "knn_preservation": knn_preservation(X, selected, n_neighbors=10),
                        "geometry_redundancy": geometry_redundancy(X, selected, random_state=seed),
                        "normalized_effective_rank": normalized_effective_rank(X, selected),
                        "fit_seconds": elapsed,
                        "ranking_depends_on_k": k_dependent,
                        "implementation_source": source,
                        "implementation_kind": _implementation_kind(source),
                        "error": "",
                    })
    return pd.DataFrame(rows)
