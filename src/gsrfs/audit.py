from __future__ import annotations

import numpy as np
import pandas as pd


def label_blindness_audit(method_factories, X, y, trials=5, random_state=1729):
    """Empirically audit that selectors do not alter output when y is permuted.

    This cannot prove absence of every possible implementation bug, but it catches
    direct or indirect use of the supplied label vector in the fitted ranking.
    Each factory must return a fresh selector with deterministic random_state.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    rows = []

    for name, factory in method_factories.items():
        base = factory()
        base.fit(X, y)
        ref = tuple(base.get_support(indices=True).tolist())
        identical = True
        for t in range(int(trials)):
            yp = rng.permutation(y)
            est = factory()
            est.fit(X, yp)
            current = tuple(est.get_support(indices=True).tolist())
            same = current == ref
            identical = identical and same
            rows.append({
                "method": name,
                "trial": t,
                "same_selection": bool(same),
                "reference_selection": ",".join(map(str, ref)),
                "permuted_selection": ",".join(map(str, current)),
            })
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method", as_index=False)["same_selection"]
        .agg(["all", "mean"])
        .reset_index()
        .rename(columns={"all": "all_trials_identical", "mean": "fraction_identical"})
    )
    return df, summary
