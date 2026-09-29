import numpy as np
from gsrfs import GSRSelector, GSRSubstitutionSelector, contextual_geometry_substitution, substitution_aware_jaccard
from gsrfs.synthetic import redundant_blocks


def kwargs(seed=0):
    return dict(
        n_features=2,
        n_permutations=6,
        max_pairs=2500,
        support_engine='vectorized',
        residual_engine='batched_cd',
        random_state=seed,
    )


def test_substitution_selector_preserves_frozen_selection():
    X, _, _, _ = redundant_blocks(seed=3, n=140, noise_features=8)
    a = GSRSelector(**kwargs(3)).fit(X)
    b = GSRSubstitutionSelector(substitution_threshold=0.9, **kwargs(3)).fit(X)
    assert np.array_equal(a.get_support(indices=True), b.get_support(indices=True))
    assert np.allclose(a.effective_support_, b.effective_support_)


def test_redundant_block_contains_true_substitute_at_reasonable_threshold():
    X, _, groups, _ = redundant_blocks(seed=1, n=180, noise_features=8)
    est = GSRSubstitutionSelector(substitution_threshold=0.90, **kwargs(1)).fit(X)
    discovered = est.get_substitution_groups()
    # Every selected signal representative should recover its paired near-copy.
    for rep in est.get_support(indices=True):
        true_group = next((g for g in groups if int(rep) in g), None)
        if true_group is not None:
            assert set(true_group).issubset(set(discovered[int(rep)]))


def test_substitution_aware_jaccard_treats_redundant_swaps_as_same_information():
    groups = {0: [0, 2], 1: [1, 3]}
    exact_a = [0, 1]
    exact_b = [2, 3]
    score = substitution_aware_jaccard(exact_a, exact_b, groups, p=8)
    assert score == 1.0


def test_contextual_substitution_self_is_one():
    H = np.eye(4)
    assert contextual_geometry_substitution(H, [0, 1], 0, 0) == 1.0


def test_grouping_is_label_blind():
    X, y, _, _ = redundant_blocks(seed=4, n=140, noise_features=8)
    est1 = GSRSubstitutionSelector(substitution_threshold=0.9, **kwargs(4)).fit(X, y=y)
    est2 = GSRSubstitutionSelector(substitution_threshold=0.9, **kwargs(4)).fit(X, y=y[::-1])
    assert np.array_equal(est1.get_support(indices=True), est2.get_support(indices=True))
    assert est1.get_substitution_groups() == est2.get_substitution_groups()


def test_substitution_aware_stability_utility_runs():
    from gsrfs import substitution_aware_stability
    X, _, _, _ = redundant_blocks(seed=7, n=120, noise_features=6)
    ref = GSRSubstitutionSelector(substitution_threshold=0.85, **kwargs(7)).fit(X)
    base = GSRSelector(**kwargs(7))
    out = substitution_aware_stability(base, X, ref.get_substitution_groups(), n_repeats=3, random_state=9)
    assert out["mean_substitution_aware_jaccard"] >= out["mean_exact_jaccard"]
