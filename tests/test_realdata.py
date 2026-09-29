from gsrfs.realdata import load_builtin_benchmarks, default_k_grid
from gsrfs.metrics import knn_preservation


def test_builtin_dataset_registry():
    datasets = load_builtin_benchmarks(include_digits=False)
    assert [d.name for d in datasets] == ["iris", "wine", "breast_cancer"]
    assert all(d.X.shape[0] == d.y.shape[0] for d in datasets)


def test_k_grid_valid():
    assert default_k_grid(4, 3) == [3, 4]


def test_knn_preservation_full_features_is_one():
    ds = load_builtin_benchmarks(include_digits=False)[0]
    score = knn_preservation(ds.X, list(range(ds.X.shape[1])), n_neighbors=5)
    assert abs(score - 1.0) < 1e-12


def test_diversity_metrics_behave_on_duplicate_features():
    import numpy as np
    from gsrfs.metrics import geometry_redundancy, normalized_effective_rank
    rng = np.random.default_rng(4)
    a = rng.normal(size=120)
    b = rng.normal(size=120)
    X = np.column_stack([a, a.copy(), b])
    red_dup = geometry_redundancy(X, [0, 1])
    red_div = geometry_redundancy(X, [0, 2])
    assert red_dup > red_div
    assert normalized_effective_rank(X, [0, 1]) < normalized_effective_rank(X, [0, 2])
