"""Optional bridges to external/canonical feature-selection implementations.

No third-party implementation source is vendored in GSR-FS.  These adapters call
an independently installed package and expose the resulting ranking through a
small scikit-learn-compatible selector interface.  This separation is deliberate:
publication benchmarks should be able to distinguish native development
reimplementations from externally sourced reference implementations.
"""
from __future__ import annotations

import importlib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class _ExternalTopK(BaseEstimator, TransformerMixin):
    ranking_depends_on_k = False

    def _set_ranking(self, X, ranking, scores=None, source="external"):
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        k = int(self.n_features)
        if not 1 <= k <= p:
            raise ValueError("n_features must lie in [1, p].")
        ranking = np.asarray(ranking, dtype=int).ravel()
        if ranking.size != p or set(ranking.tolist()) != set(range(p)):
            raise ValueError("External ranking must be a permutation of 0..p-1.")
        self.n_features_in_ = p
        self.ranking_indices_ = ranking.copy()
        self.selected_indices_ = ranking[:k].copy()
        self.support_ = np.zeros(p, dtype=bool)
        self.support_[self.selected_indices_] = True
        self.scores_ = None if scores is None else np.asarray(scores, dtype=float)
        self.external_source_ = str(source)
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        check_is_fitted(self, "support_")
        return self.selected_indices_.copy() if indices else self.support_.copy()


def _require(module):
    try:
        return importlib.import_module(module)
    except Exception as exc:
        raise ImportError(
            "This canonical bridge requires an independently installed "
            "scikit-feature package exposing the 'skfeature' namespace. "
            "GSR-FS intentionally does not vendor that GPL-licensed source."
        ) from exc


class ScikitFeatureLaplacianSelector(_ExternalTopK):
    """Bridge to scikit-feature's Laplacian Score implementation.

    The external function is called with its own default affinity construction;
    GSR-FS does not silently substitute the local development graph.
    """

    def __init__(self, n_features=10):
        self.n_features = n_features

    def fit(self, X, y=None):
        mod = _require("skfeature.function.similarity_based.lap_score")
        util = _require("skfeature.utility.construct_W")
        Xc = np.asarray(X, dtype=float).copy()
        # Upstream lap_score.py currently computes a default W when absent but
        # subsequently reads kwargs["W"].  Supplying the upstream default graph
        # explicitly executes the intended author-code path without modifying it.
        W = util.construct_W(Xc.copy())
        scores = np.asarray(mod.lap_score(Xc, W=W), dtype=float)
        ranking = np.asarray(mod.feature_ranking(scores), dtype=int)
        return self._set_ranking(
            X, ranking, scores=scores,
            source="scikit-feature:similarity_based/lap_score.py+upstream_default_W_workaround",
        )


class ScikitFeatureSPECSelector(_ExternalTopK):
    """Bridge to scikit-feature SPEC with the external implementation's defaults."""

    def __init__(self, n_features=10, style=0):
        self.n_features = n_features
        self.style = style

    def fit(self, X, y=None):
        mod = _require("skfeature.function.similarity_based.SPEC")
        Xc = np.asarray(X, dtype=float).copy()
        scores = np.asarray(mod.spec(Xc, style=int(self.style)), dtype=float)
        ranking = np.asarray(mod.feature_ranking(scores, style=int(self.style)), dtype=int)
        return self._set_ranking(
            X, ranking, scores=scores,
            source="scikit-feature:similarity_based/SPEC.py",
        )


class ScikitFeatureMCFSSelector(_ExternalTopK):
    """Bridge to scikit-feature MCFS using a label-blind fixed cluster count.

    The original scikit-feature implementation defaults to ``n_clusters=5``.
    Publication use of this bridge must keep ``n_clusters`` independent of the
    ground-truth target.  MCFS uses ``n_selected_features`` inside its LARS
    optimization, so its ranking is marked as k-dependent and must be re-fit for
    each requested k.
    """

    ranking_depends_on_k = True

    def __init__(self, n_features=10, n_clusters=5):
        self.n_features = n_features
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        mod = _require("skfeature.function.sparse_learning_based.MCFS")
        Xc = np.asarray(X, dtype=float).copy()
        kwargs = {}
        if self.n_clusters is not None:
            kwargs["n_clusters"] = int(self.n_clusters)
        weights = np.asarray(
            mod.mcfs(Xc, n_selected_features=int(self.n_features), **kwargs),
            dtype=float,
        )
        ranking = np.asarray(mod.feature_ranking(weights), dtype=int)
        # A scalar score is useful for diagnostics only; ranking comes directly
        # from the external implementation.
        scores = np.max(weights, axis=1) if weights.ndim == 2 else None
        return self._set_ranking(
            X, ranking, scores=scores,
            source="scikit-feature:sparse_learning_based/MCFS.py",
        )
