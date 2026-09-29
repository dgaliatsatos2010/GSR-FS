from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

from gsrfs import GSRSelector


def make_data(n, p, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    # Inject a few redundant structured dimensions without labels.
    if p >= 6:
        X[:, 2] = X[:, 0] + 0.05 * rng.normal(size=n)
        X[:, 3] = X[:, 1] + 0.05 * rng.normal(size=n)
        X[:, 5] = 0.7 * X[:, 4] + 0.3 * rng.normal(size=n)
    return X


def main():
    configs = [
        (120, 12, 2500),
        (250, 20, 5000),
        (500, 30, 8000),
        (800, 40, 10000),
    ]
    rows = []
    for n, p, max_pairs in configs:
        X = make_data(n, p, seed=42)
        k = min(8, p)
        engines = [
            ("reference", "nnls"),
            ("vectorized", "nnls"),
            ("vectorized", "batched_cd"),
        ]
        fitted = {}
        for support_engine, residual_engine in engines:
            est = GSRSelector(
                n_features=k,
                n_permutations=30,
                max_pairs=max_pairs,
                support_engine=support_engine,
                residual_engine=residual_engine,
                random_state=123,
            )
            t0 = time.perf_counter()
            est.fit(X)
            elapsed = time.perf_counter() - t0
            key = f"{support_engine}+{residual_engine}"
            fitted[key] = est
            rows.append({
                "n": n,
                "p": p,
                "max_pairs": max_pairs,
                "engine": key,
                "fit_seconds": elapsed,
                "selected": ",".join(map(str, est.get_support(indices=True))),
            })

        ref = fitted["reference+nnls"]
        ref_sel = ref.get_support(indices=True)
        ref_scores = ref.feature_scores_
        for row in rows[-len(engines):]:
            cur = fitted[row["engine"]]
            row["same_selection_as_reference"] = bool(
                np.array_equal(ref_sel, cur.get_support(indices=True))
            )
            row["max_abs_score_diff_vs_reference"] = float(
                np.max(np.abs(ref_scores - cur.feature_scores_))
            )

    df = pd.DataFrame(rows)
    ref_time = df[df.engine == "reference+nnls"].set_index(["n", "p"])["fit_seconds"]
    speedup = []
    for _, row in df.iterrows():
        speedup.append(float(ref_time.loc[(row.n, row.p)] / row.fit_seconds))
    df["speedup_vs_reference"] = speedup

    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "scalability_v04.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
