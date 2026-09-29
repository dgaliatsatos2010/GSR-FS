import numpy as np
import pytest
from gsrfs import GSRPartitionSelector, persistent_balanced_gap_evidence


def test_pbge_bimodal_exceeds_smooth_example():
    rng=np.random.default_rng(2); n=240
    bim=np.r_[rng.normal(-2,.25,n//2),rng.normal(2,.25,n//2)]
    smooth=rng.normal(size=n)
    assert persistent_balanced_gap_evidence(bim) > persistent_balanced_gap_evidence(smooth)


def test_partition_selector_is_label_blind():
    rng=np.random.default_rng(3)
    z=rng.integers(0,2,180)
    X=np.column_stack([np.where(z==0,-2,2)+rng.normal(0,.3,180),rng.normal(size=(180,7))])
    a=GSRPartitionSelector(n_features=2,n_permutations=8,max_pairs=2500,random_state=7).fit(X,y=z)
    b=GSRPartitionSelector(n_features=2,n_permutations=8,max_pairs=2500,random_state=7).fit(X,y=rng.permutation(z))
    assert np.array_equal(a.selected_indices_, b.selected_indices_)
    assert np.allclose(a.feature_scores_, b.feature_scores_)


def test_partition_selector_rejects_auto_v09():
    X=np.random.default_rng(1).normal(size=(50,5))
    with pytest.raises(ValueError):
        GSRPartitionSelector(n_features='auto').fit(X)
