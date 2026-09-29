import types
import numpy as np
import pytest

from gsrfs.canonical import (
    ScikitFeatureLaplacianSelector,
    ScikitFeatureSPECSelector,
    ScikitFeatureMCFSSelector,
)
from gsrfs.realdata import RealDataset
from gsrfs.benchmark import run_real_benchmark_flexible


def test_canonical_bridge_missing_dependency_message(monkeypatch):
    import gsrfs.canonical as c
    def boom(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(c.importlib, "import_module", boom)
    X = np.arange(40, dtype=float).reshape(10, 4)
    with pytest.raises(ImportError, match="independently installed"):
        ScikitFeatureLaplacianSelector(n_features=2).fit(X)


def test_canonical_laplacian_bridge_uses_external_ranking(monkeypatch):
    import gsrfs.canonical as c
    fake = types.SimpleNamespace(
        construct_W=lambda X: np.eye(X.shape[0]),
        lap_score=lambda X, **kwargs: np.array([3.0, 1.0, 2.0, 0.5]),
        feature_ranking=lambda score: np.array([3, 1, 2, 0]),
    )
    monkeypatch.setattr(c.importlib, "import_module", lambda name: fake)
    X = np.arange(40, dtype=float).reshape(10, 4)
    sel = ScikitFeatureLaplacianSelector(n_features=2).fit(X)
    assert sel.get_support(indices=True).tolist() == [3, 1]
    assert "scikit-feature" in sel.external_source_


def test_canonical_mcfs_is_marked_k_dependent(monkeypatch):
    import gsrfs.canonical as c
    calls = []
    def mcfs(X, n_selected_features, **kwargs):
        calls.append((n_selected_features, kwargs.copy()))
        w = np.zeros((X.shape[1], 2))
        w[:, 0] = np.arange(X.shape[1])
        return w
    fake = types.SimpleNamespace(
        mcfs=mcfs,
        feature_ranking=lambda W: np.argsort(np.max(W, axis=1))[::-1],
    )
    monkeypatch.setattr(c.importlib, "import_module", lambda name: fake)
    X = np.arange(50, dtype=float).reshape(10, 5)
    sel = ScikitFeatureMCFSSelector(n_features=3, n_clusters=5).fit(X)
    assert sel.ranking_depends_on_k is True
    assert calls == [(3, {"n_clusters": 5})]
    assert sel.get_support(indices=True).tolist() == [4, 3, 2]


def test_flexible_benchmark_refits_k_dependent_methods():
    class StableRank:
        ranking_depends_on_k = False
        fit_calls = 0
        def __init__(self, k): self.k = k
        def fit(self, X, y=None):
            type(self).fit_calls += 1
            self.order = np.arange(X.shape[1])
            return self
        def get_support(self, indices=False): return self.order[:self.k]

    class KDependent:
        ranking_depends_on_k = True
        fit_calls = 0
        def __init__(self, k): self.k = k
        def fit(self, X, y=None):
            type(self).fit_calls += 1
            self.order = np.arange(X.shape[1])[::-1]
            return self
        def get_support(self, indices=False): return self.order[:self.k]

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = np.repeat([0, 1, 2], 20)
    ds = RealDataset("toy", X, y, tuple(f"x{i}" for i in range(8)))
    factories = {
        "stable": lambda k, seed: StableRank(k),
        "dep": lambda k, seed: KDependent(k),
    }
    df = run_real_benchmark_flexible([ds], factories, seeds=(0,))
    ks = sorted(df.k.unique())
    assert StableRank.fit_calls == 1
    assert KDependent.fit_calls == len(ks)
    assert df.error.eq("").all()
