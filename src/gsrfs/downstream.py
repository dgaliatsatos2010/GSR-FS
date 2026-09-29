from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

from .selector import GSRSelector
from .baselines import MaxVarianceSelector, LaplacianScoreSelector, SPECSelector, MCFSSelector
from .realdata import downstream_k


def offline_method_factories(*, gsr_permutations=12, max_pairs=8000):
    """Frozen development comparators for fold-wise downstream validation.

    These are local/development baselines, not canonical third-party implementations.
    No factory accepts y or a ground-truth class count.
    """
    return {
        "GSR-FS": lambda k, seed: GSRSelector(
            n_features=k,
            n_permutations=int(gsr_permutations),
            max_pairs=int(max_pairs),
            pair_sampler="balanced",
            random_state=int(seed),
            support_engine="vectorized",
            residual_engine="auto",
        ),
        "MaxVariance": lambda k, seed: MaxVarianceSelector(n_features=k),
        "LaplacianScore-dev": lambda k, seed: LaplacianScoreSelector(n_features=k, n_neighbors=7),
        "SPEC-dev": lambda k, seed: SPECSelector(n_features=k, gamma=1.0, style=0, standardize=False),
        "MCFS-eigengap-dev": lambda k, seed: MCFSSelector(
            n_features=k,
            n_clusters="eigengap",
            n_neighbors=7,
            alpha=0.003,
            max_clusters=10,
            random_state=int(seed),
        ),
    }


def canonical_offline_method_factories(*, gsr_permutations=12, max_pairs=8000):
    """Canonical scikit-feature comparators for fold-wise validation.

    The scikit-feature package/source must be supplied independently.  No labels
    or ground-truth class counts are passed to any feature selector.  MCFS uses
    the upstream default label-blind ``n_clusters=5``.
    """
    from .canonical import (
        ScikitFeatureLaplacianSelector,
        ScikitFeatureSPECSelector,
        ScikitFeatureMCFSSelector,
    )
    return {
        "GSR-FS": lambda k, seed: GSRSelector(
            n_features=k,
            n_permutations=int(gsr_permutations),
            max_pairs=int(max_pairs),
            pair_sampler="balanced",
            random_state=int(seed),
            support_engine="vectorized",
            residual_engine="auto",
        ),
        "MaxVariance": lambda k, seed: MaxVarianceSelector(n_features=k),
        "LaplacianScore-scikit-feature": lambda k, seed: ScikitFeatureLaplacianSelector(n_features=k),
        "SPEC-scikit-feature": lambda k, seed: ScikitFeatureSPECSelector(n_features=k, style=0),
        "MCFS-scikit-feature-k5": lambda k, seed: ScikitFeatureMCFSSelector(
            n_features=k, n_clusters=5
        ),
    }


def _splitter_for_dataset(ds, n_splits=3, seed=0):
    if ds.groups is not None:
        if StratifiedGroupKFold is None:
            raise RuntimeError("StratifiedGroupKFold is unavailable in this scikit-learn version.")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return splitter.split(ds.X, ds.y, groups=ds.groups), "stratified_group"
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(ds.X, ds.y), "stratified"


def run_foldwise_downstream_validation(
    datasets,
    *,
    method_factories=None,
    seeds=(0,),
    n_splits=3,
    gsr_permutations=12,
    max_pairs=8000,
):
    """Evaluate selected features without transductive feature-selection leakage.

    For every fold, the selector is fit on X_train only and never receives y.
    A standard scaler and logistic-regression classifier are then fit on the
    selected training features. Labels are used only by the splitter and the
    downstream classifier/evaluation, not by feature selection.
    """
    method_factories = method_factories or offline_method_factories(
        gsr_permutations=gsr_permutations, max_pairs=max_pairs
    )
    rows = []
    for ds in datasets:
        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y)
        k = downstream_k(X.shape[1])
        for seed in seeds:
            split_iter, split_kind = _splitter_for_dataset(ds, n_splits=n_splits, seed=int(seed))
            splits = list(split_iter)
            for fold, (tr, te) in enumerate(splits):
                Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
                for method_name, factory in method_factories.items():
                    selector = factory(k, int(seed) * 1000 + fold)
                    t0 = time.perf_counter()
                    try:
                        selector.fit(Xtr)
                        selected = np.asarray(selector.get_support(indices=True), dtype=int)
                        fit_seconds = time.perf_counter() - t0
                        implementation_source = getattr(selector, "external_source_", "native_or_local")
                        if selected.size == 0:
                            raise RuntimeError("selector returned an empty fixed-k subset")
                        clf = make_pipeline(
                            StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced"),
                        )
                        clf.fit(Xtr[:, selected], ytr)
                        pred = clf.predict(Xte[:, selected])
                        rows.append({
                            "dataset": ds.name,
                            "panel_status": ds.panel_status,
                            "source": ds.source,
                            "seed": int(seed),
                            "fold": int(fold),
                            "split_kind": split_kind,
                            "method": method_name,
                            "implementation_source": implementation_source,
                            "n_samples": X.shape[0],
                            "n_features": X.shape[1],
                            "n_classes": len(np.unique(y)),
                            "k": int(k),
                            "selected": ",".join(map(str, selected.tolist())),
                            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
                            "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
                            "accuracy": float(accuracy_score(yte, pred)),
                            "selector_fit_seconds": float(fit_seconds),
                            "error": "",
                        })
                    except Exception as exc:
                        rows.append({
                            "dataset": ds.name,
                            "panel_status": ds.panel_status,
                            "source": ds.source,
                            "seed": int(seed),
                            "fold": int(fold),
                            "split_kind": split_kind,
                            "method": method_name,
                            "implementation_source": (
                                "scikit-feature:execution_failed"
                                if "scikit-feature" in method_name
                                else "native_or_local"
                            ),
                            "n_samples": X.shape[0],
                            "n_features": X.shape[1],
                            "n_classes": len(np.unique(y)),
                            "k": int(k),
                            "selected": "",
                            "balanced_accuracy": np.nan,
                            "macro_f1": np.nan,
                            "accuracy": np.nan,
                            "selector_fit_seconds": float(time.perf_counter() - t0),
                            "error": repr(exc),
                        })
    return pd.DataFrame(rows)


def summarize_foldwise_downstream(df):
    metrics = ["balanced_accuracy", "macro_f1", "accuracy", "selector_fit_seconds"]
    per_dataset = (
        df.groupby(["dataset", "panel_status", "method"], dropna=False)[metrics]
        .mean()
        .reset_index()
    )
    overall = (
        per_dataset.groupby("method", dropna=False)[metrics]
        .agg(["mean", "std"])
        .sort_values(("balanced_accuracy", "mean"), ascending=False)
    )
    extension = (
        per_dataset.loc[per_dataset["panel_status"].eq("offline_extension")]
        .groupby("method", dropna=False)[metrics]
        .agg(["mean", "std"])
        .sort_values(("balanced_accuracy", "mean"), ascending=False)
    )
    return overall, extension
