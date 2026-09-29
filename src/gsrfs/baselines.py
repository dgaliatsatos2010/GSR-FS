from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors


class _TopKSelector(BaseEstimator, TransformerMixin):
    def _finish(self, X, scores, k, higher_is_better=True):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        k = int(k)
        if not 1 <= k <= self.n_features_in_:
            raise ValueError("n_features must lie in [1, p].")
        order = np.argsort(-scores if higher_is_better else scores)
        self.selected_indices_ = np.asarray(order[:k], dtype=int)
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[self.selected_indices_] = True
        self.scores_ = np.asarray(scores, dtype=float)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]

    def get_support(self, indices=False):
        return self.selected_indices_.copy() if indices else self.support_.copy()


def _standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (X - mu) / sd


def _heat_knn_graph(Z, n_neighbors=7):
    n = Z.shape[0]
    k = min(max(int(n_neighbors) + 1, 2), n)
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    dist, ind = nn.kneighbors(Z)
    d2 = dist[:, 1:] ** 2
    nbr = ind[:, 1:]
    positive = d2[d2 > 0]
    t = float(np.median(positive)) if positive.size else 1.0
    weights = np.exp(-d2 / max(t, 1e-12))
    rows = np.repeat(np.arange(n), nbr.shape[1])
    W = sparse.coo_matrix(
        (weights.ravel(), (rows, nbr.ravel())), shape=(n, n)
    ).tocsr()
    W = W.maximum(W.T)
    return W


def estimate_spectral_cluster_count(
    X,
    n_neighbors=7,
    min_clusters=2,
    max_clusters=10,
):
    """Estimate a cluster-count surrogate from X alone using an eigengap rule.

    This helper exists to prevent a subtle but important form of label leakage:
    methods such as MCFS are often configured with the *ground-truth* class count.
    For a strict unsupervised benchmark that is not allowed.  The estimate here
    depends only on the predictor geometry.
    """
    Z = _standardize(X)
    n = Z.shape[0]
    if n < 4:
        return 2
    W = _heat_knn_graph(Z, n_neighbors=n_neighbors)
    d = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Lsym = sparse.eye(n, format="csr") - sparse.diags(dinv) @ W @ sparse.diags(dinv)

    upper = min(int(max_clusters), n - 2)
    lower = min(max(int(min_clusters), 2), upper)
    if upper < 2:
        return 2

    # Need lambda_0 ... lambda_upper to score the gap after k small eigenvalues.
    nev = min(upper + 1, n - 1)
    vals = eigsh(Lsym, k=nev, which="SM", return_eigenvectors=False)
    vals = np.sort(np.real(vals))
    candidates = np.arange(lower, min(upper, len(vals) - 1) + 1)
    if candidates.size == 0:
        return 2
    gaps = np.asarray([vals[k] - vals[k - 1] for k in candidates], dtype=float)
    return int(candidates[int(np.argmax(gaps))])


class MaxVarianceSelector(_TopKSelector):
    def __init__(self, n_features=10):
        self.n_features = n_features

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        return self._finish(X, np.var(X, axis=0), self.n_features, True)


class LaplacianScoreSelector(_TopKSelector):
    """Laplacian Score-style local reimplementation for development benchmarking."""

    def __init__(self, n_features=10, n_neighbors=5, t=None):
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.t = t

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Z = _standardize(X)
        n = Z.shape[0]
        k = min(int(self.n_neighbors) + 1, n)
        nn = NearestNeighbors(n_neighbors=k).fit(Z)
        dist, ind = nn.kneighbors(Z)
        d2 = dist[:, 1:] ** 2
        nbr = ind[:, 1:]
        if self.t is None:
            positive = d2[d2 > 0]
            t = float(np.median(positive)) if positive.size else 1.0
        else:
            t = float(self.t)
        weights = np.exp(-d2 / max(t, 1e-12))
        rows = np.repeat(np.arange(n), nbr.shape[1])
        W = sparse.coo_matrix((weights.ravel(), (rows, nbr.ravel())), shape=(n, n)).tocsr()
        W = W.maximum(W.T)
        d = np.asarray(W.sum(axis=1)).ravel()
        L = sparse.diags(d) - W
        dsum = d.sum() + 1e-12

        scores = np.empty(Z.shape[1])
        for j in range(Z.shape[1]):
            f = Z[:, j]
            f = f - (f @ d) / dsum
            denom = float(f @ (d * f))
            scores[j] = float(f @ (L @ f)) / (denom + 1e-12)
        return self._finish(X, scores, self.n_features, False)


class SPECSelector(_TopKSelector):
    """SPEC-style spectral feature ranking (ICML 2007), local reimplementation.

    The default ``style=0`` follows the commonly used all-nontrivial-eigenmodes
    scoring family.  This is a clean-room local adapter for comparative research,
    not a claim of bitwise identity with scikit-feature.
    """

    def __init__(self, n_features=10, gamma=1.0, style=0, standardize=False):
        self.n_features = n_features
        self.gamma = gamma
        self.style = style
        self.standardize = standardize

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Z = _standardize(X) if bool(self.standardize) else X.copy()
        n, p = Z.shape
        W = rbf_kernel(Z, gamma=float(self.gamma))
        d = W.sum(axis=1)
        d_safe = np.maximum(d, 1e-12)
        dinv = 1.0 / np.sqrt(d_safe)
        L = np.diag(d) - W
        Lhat = (dinv[:, None] * L) * dinv[None, :]
        vals, U = np.linalg.eigh(Lhat)
        # scikit-feature's implementation reverses both arrays; reproduce the
        # same algebraic ordering without copying its source code.
        vals = vals[::-1]
        U = U[:, ::-1]
        v = np.sqrt(d_safe)
        v = v / (np.linalg.norm(v) + 1e-12)

        scores = np.full(p, -np.inf, dtype=float)
        for j in range(p):
            fhat = np.sqrt(d_safe) * Z[:, j]
            norm = np.linalg.norm(fhat)
            if norm <= 100 * np.spacing(1):
                continue
            fhat = fhat / norm
            a = (fhat @ U) ** 2
            if int(self.style) == -1:
                score = float(np.sum(a * vals))
            elif int(self.style) == 0:
                denom = 1.0 - float((fhat @ v) ** 2)
                score = float(np.sum(a[:-1] * vals[:-1]) / (denom + 1e-12))
            else:
                kk = min(max(int(self.style), 2), n)
                score = float(np.sum(a[n-kk:n-1] * (2.0 - vals[n-kk:n-1])))
                # For this style lower is conventionally better; negate so the
                # common _finish path can still treat higher as better.
                score = -score
            scores[j] = score
        return self._finish(X, scores, self.n_features, True)


class MCFSSelector(_TopKSelector):
    """Compact MCFS development baseline with strictly label-blind dimension choice.

    ``n_clusters='eigengap'`` is the v0.5 default.  It estimates the spectral
    embedding dimension from X only.  Supplying an integer is allowed for
    sensitivity studies, but publication scripts must not derive it from y.
    """

    def __init__(
        self,
        n_features=10,
        n_clusters="eigengap",
        n_neighbors=5,
        alpha=0.01,
        max_clusters=10,
        random_state=None,
    ):
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.max_clusters = max_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Z = _standardize(X)
        n = Z.shape[0]

        if self.n_clusters in (None, "eigengap"):
            n_clusters = estimate_spectral_cluster_count(
                Z,
                n_neighbors=self.n_neighbors,
                max_clusters=self.max_clusters,
            )
        else:
            n_clusters = int(self.n_clusters)
            if n_clusters < 2:
                raise ValueError("n_clusters must be >=2 or 'eigengap'.")
        self.n_clusters_estimated_ = int(n_clusters)

        W = _heat_knn_graph(Z, n_neighbors=self.n_neighbors)
        d = np.asarray(W.sum(axis=1)).ravel()
        dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        Lsym = sparse.eye(n) - sparse.diags(dinv) @ W @ sparse.diags(dinv)

        nvec = min(max(int(n_clusters) + 1, 2), n - 1)
        vals, vecs = eigsh(Lsym, k=nvec, which="SM")
        order = np.argsort(vals)
        Y = vecs[:, order[1:min(nvec, int(n_clusters) + 1)]]

        coef = []
        for c in range(Y.shape[1]):
            model = Lasso(
                alpha=float(self.alpha),
                fit_intercept=True,
                max_iter=5000,
                random_state=self.random_state,
            )
            model.fit(Z, Y[:, c])
            coef.append(np.abs(model.coef_))
        scores = np.max(np.vstack(coef), axis=0) if coef else np.var(Z, axis=0)
        return self._finish(X, scores, self.n_features, True)


class ExternalRankingSelector(_TopKSelector):
    """Adapter for rankings produced by canonical/author implementations.

    This avoids copying third-party source into GSR-FS while still letting the
    benchmark consume a verified ranking generated externally (Python/MATLAB/R).
    ``ranking`` must list feature indices from most to least preferred.
    """

    def __init__(self, ranking, n_features=10, source="external"):
        self.ranking = ranking
        self.n_features = n_features
        self.source = source

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        ranking = np.asarray(self.ranking, dtype=int)
        if ranking.ndim != 1 or len(ranking) != p:
            raise ValueError("ranking must contain exactly p feature indices.")
        if set(ranking.tolist()) != set(range(p)):
            raise ValueError("ranking must be a permutation of 0..p-1.")
        # Encode rank as descending score solely to reuse _TopKSelector.
        scores = np.empty(p, dtype=float)
        scores[ranking] = np.arange(p, 0, -1, dtype=float)
        self.external_source_ = str(self.source)
        return self._finish(X, scores, self.n_features, True)
