import numpy as np
import pytest

from gsrfs import GSRSelector, GSRConcordanceSelector, jackknife_geometry_concordance


def _data(seed=0):
    rng = np.random.default_rng(seed)
    n = 100
    z1 = np.r_[rng.normal(-2, .5, n//2), rng.normal(2, .5, n//2)]
    z2 = np.r_[rng.normal(-1, .6, n//2), rng.normal(1, .6, n//2)]
    return np.column_stack([z1, z2, z1+rng.normal(0,.03,n), z2+rng.normal(0,.03,n), rng.normal(size=(n,6))])


def test_concordance_fixed_k_and_label_blind():
    X = _data()
    y1 = np.arange(len(X)) % 2
    y2 = np.arange(len(X)) % 3
    a = GSRConcordanceSelector(n_features=2, n_permutations=5, max_pairs=1500, n_blocks=4, random_state=3).fit(X, y1)
    b = GSRConcordanceSelector(n_features=2, n_permutations=5, max_pairs=1500, n_blocks=4, random_state=3).fit(X, y2)
    assert np.array_equal(a.get_support(indices=True), b.get_support(indices=True))
    assert a.jgc_fold_ranks_.shape == (4, X.shape[1])
    assert np.all((a.jgc_lower_rank_ >= 0) & (a.jgc_lower_rank_ <= 1))


def test_concordance_rejects_auto():
    X = _data()
    with pytest.raises(ValueError):
        GSRConcordanceSelector(n_features='auto').fit(X)


def test_zero_tie_tolerance_matches_base_selection_path():
    X = _data(2)
    kw = dict(n_features=2, n_permutations=5, max_pairs=1500, random_state=9)
    base = GSRSelector(**kw).fit(X)
    jgc = GSRConcordanceSelector(tie_tolerance=0.0, n_blocks=4, **kw).fit(X)
    assert np.array_equal(base.get_support(indices=True), jgc.get_support(indices=True))
