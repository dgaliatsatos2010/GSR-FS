import numpy as np
from sklearn.datasets import load_iris

from gsrfs import (
    GSRSelector, MaxVarianceSelector, LaplacianScoreSelector,
    SPECSelector, MCFSSelector, estimate_spectral_cluster_count,
    label_blindness_audit,
)


def test_spectral_cluster_count_is_x_only_and_valid():
    X, y = load_iris(return_X_y=True)
    k1 = estimate_spectral_cluster_count(X, n_neighbors=7, max_clusters=8)
    # y is deliberately irrelevant; only X is passed.
    k2 = estimate_spectral_cluster_count(X.copy(), n_neighbors=7, max_clusters=8)
    assert k1 == k2
    assert 2 <= k1 <= 8


def test_mcfs_ignores_y_and_uses_eigengap():
    X, y = load_iris(return_X_y=True)
    a = MCFSSelector(n_features=2, n_clusters="eigengap", random_state=0).fit(X, y)
    b = MCFSSelector(n_features=2, n_clusters="eigengap", random_state=0).fit(X, y[::-1])
    assert np.array_equal(a.get_support(indices=True), b.get_support(indices=True))
    assert a.n_clusters_estimated_ == b.n_clusters_estimated_


def test_all_bundled_publication_selectors_are_label_blind_empirically():
    X, y = load_iris(return_X_y=True)
    factories = {
        "GSR-FS": lambda: GSRSelector(n_features=2, n_permutations=10, max_pairs=3000, random_state=4),
        "MaxVariance": lambda: MaxVarianceSelector(n_features=2),
        "LaplacianScore": lambda: LaplacianScoreSelector(n_features=2, n_neighbors=7),
        "SPEC": lambda: SPECSelector(n_features=2),
        "MCFS": lambda: MCFSSelector(n_features=2, n_clusters="eigengap", random_state=4),
    }
    raw, summary = label_blindness_audit(factories, X, y, trials=3, random_state=99)
    assert raw["same_selection"].all()
    assert summary["all_trials_identical"].all()


def test_spec_returns_valid_topk():
    X, _ = load_iris(return_X_y=True)
    sel = SPECSelector(n_features=2).fit(X)
    idx = sel.get_support(indices=True)
    assert len(idx) == 2
    assert len(set(idx.tolist())) == 2
