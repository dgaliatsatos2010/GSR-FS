import numpy as np

from gsrfs import GSRSelector


def _structured(seed=0, n=140, p_noise=8):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    return np.column_stack([
        z1,
        z2,
        z1 + 0.08 * rng.normal(size=n),
        np.sin(z2) + 0.08 * rng.normal(size=n),
        rng.normal(size=(n, p_noise)),
    ])


def test_feature_order_equivariance_under_common_permutation_bank():
    X = _structured(4)
    p = X.shape[1]
    col_rng = np.random.default_rng(123)
    perm = col_rng.permutation(p)

    kw = dict(
        n_features=5,
        n_permutations=30,
        max_pairs=2500,
        random_state=77,
        support_engine="vectorized",
        residual_engine="nnls",
    )
    a = GSRSelector(**kw).fit(X)
    b = GSRSelector(**kw).fit(X[:, perm])

    mapped_path = perm[b.get_support(indices=True)]
    assert np.array_equal(a.get_support(indices=True), mapped_path)

    # Feature-wise support statistics must be identical after mapping columns back.
    mapped_support = np.empty(p)
    mapped_support[perm] = b.effective_support_
    assert np.allclose(a.effective_support_, mapped_support, rtol=1e-12, atol=1e-12)
    assert a.permutation_bank_sha256_ == b.permutation_bank_sha256_


def test_pair_budget_does_not_change_permutation_bank():
    X = _structured(9, n=180)
    common = dict(
        n_features=4,
        n_permutations=25,
        random_state=2026,
        support_engine="vectorized",
        residual_engine="nnls",
    )
    small = GSRSelector(max_pairs=800, **common).fit(X)
    large = GSRSelector(max_pairs=5000, **common).fit(X)

    assert small.permutation_bank_sha256_ == large.permutation_bank_sha256_
    assert small.rng_policy_ == "split_pair_permutation_streams_v06"
    assert large.rng_policy_ == "split_pair_permutation_streams_v06"


def test_pair_sampler_is_nested_across_budgets():
    n = 220
    base_seed = 31337

    def sampled(max_pairs):
        obj = GSRSelector(n_features=2, n_permutations=20, max_pairs=max_pairs, random_state=0)
        seed_seq = np.random.SeedSequence(base_seed)
        pair_seed, _ = seed_seq.spawn(2)
        rng = np.random.default_rng(pair_seed)
        i, j = obj._sample_pairs(n, rng)
        return np.column_stack([i, j])

    a = sampled(500)
    b = sampled(1000)
    c = sampled(2500)
    assert np.array_equal(a, b[:500])
    assert np.array_equal(b, c[:1000])


def test_auto_residual_engine_switches_on_tall_geometry():
    obj = GSRSelector(n_features=2, n_permutations=20, residual_engine="auto")
    assert obj._resolved_residual_engine(np.empty((5000, 20))) == "nnls"
    assert obj._resolved_residual_engine(np.empty((16000, 20))) == "batched_cd"
    assert obj._resolved_residual_engine(np.empty((5000, 60))) == "batched_cd"


def test_balanced_pair_sampler_unique_and_endpoint_balanced():
    n = 101
    obj = GSRSelector(n_features=2, n_permutations=20, max_pairs=5000, pair_sampler="balanced")
    rng = np.random.default_rng(1234)
    i, j = obj._sample_pairs(n, rng)
    assert len(i) == 5000
    pairs = set(zip(i.tolist(), j.tolist()))
    assert len(pairs) == 5000
    assert np.all(i < j)
    deg = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
    assert int(deg.max() - deg.min()) <= 2
