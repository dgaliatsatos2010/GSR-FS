import numpy as np
import pandas as pd

from gsrfs import GSRSelector
from gsrfs.stats import compare_methods, holm_adjust


def structured_X(seed=0, n=120, p=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 2] = X[:, 0] + 0.05 * rng.normal(size=n)
    X[:, 3] = X[:, 1] + 0.05 * rng.normal(size=n)
    return X


def test_vectorized_support_matches_reference():
    X = structured_X(1)
    kw = dict(n_features=5, n_permutations=25, max_pairs=2500, random_state=77)
    ref = GSRSelector(**kw, support_engine="reference", residual_engine="nnls").fit(X)
    fast = GSRSelector(**kw, support_engine="vectorized", residual_engine="nnls").fit(X)
    assert np.allclose(ref.effective_support_, fast.effective_support_, rtol=1e-8, atol=1e-9)
    assert np.array_equal(ref.get_support(indices=True), fast.get_support(indices=True))


def test_batched_residual_matches_nnls_selection_path():
    X = structured_X(2)
    kw = dict(n_features=7, n_permutations=20, max_pairs=2200, random_state=91)
    ref = GSRSelector(**kw, support_engine="vectorized", residual_engine="nnls").fit(X)
    fast = GSRSelector(**kw, support_engine="vectorized", residual_engine="batched_cd").fit(X)
    assert np.array_equal(ref.get_support(indices=True), fast.get_support(indices=True))
    assert np.allclose(ref.feature_scores_, fast.feature_scores_, rtol=1e-6, atol=1e-8)


def test_holm_adjust_monotone_and_bounded():
    out = holm_adjust([0.01, 0.03, 0.20])
    assert np.all((out >= 0) & (out <= 1))
    assert out[0] <= out[1] <= out[2]


def test_multidataset_stats_reference_better():
    rows = []
    for d in range(8):
        for method, base in [("GSR-FS", 0.80), ("A", 0.60), ("B", 0.55)]:
            rows.append({"dataset": f"d{d}", "method": method, "nmi": base + 0.001*d})
    res = compare_methods(pd.DataFrame(rows), "nmi", reference="GSR-FS", higher_is_better=True)
    assert res["n_complete_datasets"] == 8
    assert res["mean_ranks"].iloc[0]["method"] == "GSR-FS"
    assert (res["posthoc"]["rank_biserial"] > 0).all()
