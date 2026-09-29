import numpy as np
import pandas as pd

from gsrfs import GSRSelector, GSRSubstitutionSelector, GSRStructuralGroupSelector
from gsrfs.synthetic import redundant_blocks


def kwargs(seed=0):
    return dict(
        n_features=2,
        n_permutations=8,
        max_pairs=2500,
        support_engine="vectorized",
        residual_engine="batched_cd",
        random_state=seed,
    )


def test_structural_group_output_preserves_frozen_selection_and_transform():
    X, _, _, _ = redundant_blocks(seed=12, n=160, noise_features=8)
    base = GSRSelector(**kwargs(12)).fit(X)
    grp = GSRStructuralGroupSelector(**kwargs(12)).fit(X)
    assert np.array_equal(base.get_support(indices=True), grp.get_support(indices=True))
    assert np.allclose(base.transform(X), grp.transform_representatives(X))
    assert grp.structural_group_output_creates_latent_features_ is False
    assert grp.structural_group_output_changes_selection_ is False


def test_group_output_recovers_redundant_members_and_has_stable_ids():
    X, _, true_groups, _ = redundant_blocks(seed=5, n=180, noise_features=8)
    est = GSRStructuralGroupSelector(substitution_threshold=0.85, **kwargs(5)).fit(X)
    groups = est.get_structural_groups()
    assert [g["group_id"] for g in groups] == ["SG01", "SG02"]
    for g in groups:
        rep = g["representative_index"]
        truth = next((tg for tg in true_groups if rep in tg), None)
        if truth is not None:
            assert set(truth).issubset(set(g["member_indices"]))
        assert g["output_feature"] == g["representative"]


def test_group_feature_map_and_report_are_human_readable():
    rng = np.random.default_rng(2)
    z = rng.normal(size=120)
    df = pd.DataFrame({
        "signal": z,
        "signal_copy": z + rng.normal(scale=0.01, size=120),
        "noise": rng.normal(size=120),
    })
    est = GSRStructuralGroupSelector(
        n_features=1, n_permutations=8, max_pairs=2000,
        support_engine="vectorized", residual_engine="batched_cd",
        substitution_threshold=0.85, random_state=2,
    ).fit(df)
    fmap = est.get_feature_group_map(names=True)
    assert fmap["signal"] is not None or fmap["signal_copy"] is not None
    report = est.get_structural_group_report()
    assert {"group_id", "representative", "members", "output_feature"}.issubset(report.columns)
    summary = est.get_group_summary()
    assert summary["creates_latent_features"] is False
    assert summary["representative_output_dimension"] == 1


def test_group_output_is_label_blind():
    X, y, _, _ = redundant_blocks(seed=19, n=150, noise_features=8)
    a = GSRStructuralGroupSelector(**kwargs(19)).fit(X, y=y)
    b = GSRStructuralGroupSelector(**kwargs(19)).fit(X, y=y[::-1])
    assert a.get_structural_groups() == b.get_structural_groups()


def test_frozen_cgs_default_is_085():
    assert GSRSubstitutionSelector().substitution_threshold == 0.85
    assert GSRStructuralGroupSelector().substitution_threshold == 0.85
