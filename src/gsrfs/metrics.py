from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def group_coverage(selected, groups):
    if not groups:
        return np.nan
    s = set(map(int, selected))
    return float(np.mean([bool(s.intersection(g)) for g in groups]))


def redundancy_rate(selected, groups):
    if not groups:
        return np.nan
    s = set(map(int, selected))
    redundant = 0
    selected_relevant = 0
    for g in groups:
        hits = len(s.intersection(g))
        selected_relevant += hits
        redundant += max(0, hits - 1)
    return float(redundant / max(selected_relevant, 1))


def relevant_precision(selected, groups):
    if not groups:
        return np.nan
    relevant = set(i for g in groups for i in g)
    s = set(map(int, selected))
    return float(len(s.intersection(relevant)) / max(len(s), 1))


def clustering_utility(X, y, selected, n_clusters, random_state=0):
    selected = np.asarray(selected, dtype=int)
    if selected.size == 0:
        return {"ari": np.nan, "nmi": np.nan, "silhouette": np.nan}
    Z = StandardScaler().fit_transform(np.asarray(X)[:, selected])
    pred = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(Z)
    sil = np.nan
    if len(np.unique(pred)) > 1:
        sil = float(silhouette_score(Z, pred))
    return {
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "silhouette": sil,
    }


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def knn_preservation(X, selected, n_neighbors=10):
    """Mean fraction of full-space nearest neighbors retained after selection.

    This is a label-free evaluation metric. Both the full and selected spaces are
    standardized before neighbor construction. A value of 1 means every sample's
    selected-space kNN set matches its full-space kNN set exactly.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    selected = np.asarray(selected, dtype=int)
    if selected.size == 0:
        return np.nan
    if selected.size == X.shape[1] and np.array_equal(np.sort(selected), np.arange(X.shape[1])):
        return 1.0
    n = X.shape[0]
    if n < 3:
        return np.nan
    k = min(int(n_neighbors), n - 1)
    if k < 1:
        return np.nan

    Zfull = StandardScaler().fit_transform(X)
    Zsel = StandardScaler().fit_transform(X[:, selected])
    full_nn = NearestNeighbors(n_neighbors=k + 1).fit(Zfull)
    sel_nn = NearestNeighbors(n_neighbors=k + 1).fit(Zsel)
    full_ind = full_nn.kneighbors(Zfull, return_distance=False)[:, 1:]
    sel_ind = sel_nn.kneighbors(Zsel, return_distance=False)[:, 1:]
    overlaps = [len(set(a).intersection(b)) / k for a, b in zip(full_ind, sel_ind)]
    return float(np.mean(overlaps))


def geometry_redundancy(X, selected, max_pairs=20000, random_state=0):
    """Mean pairwise cosine similarity among selected feature geometry profiles.

    Lower values indicate that selected features contribute less-duplicative
    pairwise geometry. This metric is fully label-free and is used only for
    evaluation, not by the selector itself.
    """
    X = np.asarray(X, dtype=float)
    selected = np.asarray(selected, dtype=int)
    if selected.size < 2:
        return 0.0
    Z = StandardScaler().fit_transform(X[:, selected])
    n = Z.shape[0]
    total = n * (n - 1) // 2
    rng = np.random.default_rng(random_state)
    if total <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        pairs = set()
        batch = max(4096, max_pairs * 2)
        while len(pairs) < max_pairs:
            a = rng.integers(0, n, size=batch)
            b = rng.integers(0, n, size=batch)
            m = a != b
            lo = np.minimum(a[m], b[m])
            hi = np.maximum(a[m], b[m])
            pairs.update(zip(lo.tolist(), hi.tolist()))
        arr = np.asarray(list(pairs)[:max_pairs], dtype=int)
        i, j = arr[:, 0], arr[:, 1]
    G = (Z[i, :] - Z[j, :]) ** 2
    G /= np.linalg.norm(G, axis=0, keepdims=True) + 1e-12
    C = G.T @ G
    tri = C[np.triu_indices(C.shape[0], 1)]
    return float(np.mean(tri)) if tri.size else 0.0


def normalized_effective_rank(X, selected):
    """Entropy effective rank divided by selected dimension; range approximately [0,1]."""
    X = np.asarray(X, dtype=float)
    selected = np.asarray(selected, dtype=int)
    if selected.size <= 1:
        return 1.0 if selected.size == 1 else np.nan
    Z = StandardScaler().fit_transform(X[:, selected])
    s = np.linalg.svd(Z, compute_uv=False)
    eig = s * s
    if eig.sum() <= 1e-12:
        return 0.0
    q = eig / eig.sum()
    erank = np.exp(-np.sum(q * np.log(q + 1e-12)))
    return float(erank / selected.size)
