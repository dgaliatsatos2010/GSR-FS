import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from gsrfs import GSRSelector


def test_dataframe_names_preserved():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(80, 5)), columns=list("abcde"))
    sel = GSRSelector(n_features=2, n_permutations=10, random_state=1).fit(X)
    assert len(sel.get_feature_names_out()) == 2
    assert set(sel.get_feature_names_out()).issubset(set(X.columns))


def test_pipeline_compatible():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(70, 6))
    pipe = Pipeline([
        ("select", GSRSelector(n_features=3, n_permutations=10, random_state=2)),
        ("scale", StandardScaler()),
    ])
    Xt = pipe.fit_transform(X)
    assert Xt.shape == (70, 3)


def test_signal_family_recovery():
    rng = np.random.default_rng(7)
    n = 180
    y = np.repeat([0, 1], n // 2)
    z1 = rng.normal(np.where(y == 0, -2, 2), 0.5)
    z2 = rng.normal(np.where(y == 0, -1.5, 1.5), 0.6)
    X = np.column_stack([
        z1, z2,
        z1 + rng.normal(0, 0.05, n),
        z2 + rng.normal(0, 0.05, n),
        rng.normal(size=(n, 16)),
    ])
    sel = GSRSelector(n_features=2, n_permutations=15, random_state=123).fit(X)
    chosen = set(sel.get_support(indices=True))
    assert chosen & {0, 2}
    assert chosen & {1, 3}


def test_no_residual_ablation_ranks_by_effective_support():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 8))
    sel = GSRSelector(
        n_features=4,
        n_permutations=10,
        use_residual=False,
        random_state=11,
    ).fit(X)
    expected = np.argsort(-sel.effective_support_)[:4]
    assert np.array_equal(sel.get_support(indices=True), expected)


def test_auto_null_can_return_empty_and_report():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 10))
    sel = GSRSelector(
        n_features="auto",
        n_permutations=50,
        alpha=0.05,
        max_pairs=2500,
        random_state=123,
    ).fit(X)
    assert sel.n_features_selected_ == 0
    report = sel.get_feature_report()
    assert len(report) == X.shape[1]
    assert {"feature", "selected", "max_null_adjusted_p"}.issubset(report.columns)
