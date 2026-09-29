import numpy as np
from gsrfs.benchmark import run_synthetic_benchmark


def test_benchmark_smoke():
    df = run_synthetic_benchmark(
        scenarios=["redundant_blocks"],
        seeds=(0,),
        gsr_permutations=5,
        max_pairs=2000,
    )
    assert set(df["method"]) == {"GSR-FS", "MaxVariance", "LaplacianScore", "SPEC", "MCFS"}
    assert df["error"].eq("").all()


def test_summarize_real_is_dataset_balanced():
    import pandas as pd
    from gsrfs.benchmark import summarize_real

    # d1 deliberately has many repeated rows; a naive row-level mean would be biased.
    rows = []
    for _ in range(10):
        rows.append({"dataset":"d1","method":"A","ari":1.0,"nmi":1.0,"silhouette":1.0,"knn_preservation":1.0,"geometry_redundancy":0.0,"normalized_effective_rank":1.0,"fit_seconds":1.0})
    rows.append({"dataset":"d2","method":"A","ari":0.0,"nmi":0.0,"silhouette":0.0,"knn_preservation":0.0,"geometry_redundancy":1.0,"normalized_effective_rank":0.0,"fit_seconds":1.0})
    for d in ["d1","d2"]:
        rows.append({"dataset":d,"method":"B","ari":0.4,"nmi":0.4,"silhouette":0.4,"knn_preservation":0.4,"geometry_redundancy":0.4,"normalized_effective_rank":0.4,"fit_seconds":1.0})
    s = summarize_real(pd.DataFrame(rows))
    assert np.isclose(s.loc["A", ("nmi", "mean")], 0.5)
