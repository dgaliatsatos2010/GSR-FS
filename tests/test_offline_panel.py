import numpy as np
from gsrfs.realdata import load_statsmodels_extension_benchmarks, load_offline_real_panel, downstream_k
from gsrfs.downstream import run_foldwise_downstream_validation


def test_statsmodels_extension_panel_is_real_and_well_formed():
    ds = load_statsmodels_extension_benchmarks()
    assert [d.name for d in ds] == ["anes96", "modechoice", "spector"]
    for d in ds:
        assert d.X.ndim == 2
        assert len(d.y) == d.X.shape[0]
        assert d.X.shape[1] >= 3
        assert len(np.unique(d.y)) >= 2
        assert d.panel_status == "offline_extension"
    assert ds[1].groups is not None
    assert len(np.unique(ds[1].groups)) < len(ds[1].groups)


def test_downstream_k_is_label_independent_dimension_rule():
    assert downstream_k(2) == 2
    assert downstream_k(3) == 2
    assert downstream_k(4) == 2
    assert downstream_k(9) == 3
    assert downstream_k(64) == 8


def test_foldwise_selector_never_requires_y():
    # Tiny use of Spector keeps this regression test fast.
    ds = [load_statsmodels_extension_benchmarks()[-1]]
    out = run_foldwise_downstream_validation(
        ds,
        seeds=(0,),
        n_splits=2,
        gsr_permutations=5,
        max_pairs=200,
    )
    assert set(out["method"]) >= {"GSR-FS", "MaxVariance"}
    assert out["error"].fillna("").eq("").all()
    assert out["balanced_accuracy"].notna().all()
