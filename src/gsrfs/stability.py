from __future__ import annotations

import numpy as np
from sklearn.base import clone
from .metrics import jaccard


def subsample_stability(selector, X, n_repeats=10, sample_fraction=0.8, random_state=0):
    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    selections = []
    for r in range(int(n_repeats)):
        idx = rng.choice(X.shape[0], size=max(3, int(round(sample_fraction * X.shape[0]))), replace=False)
        est = clone(selector)
        if hasattr(est, "random_state"):
            est.set_params(random_state=random_state + r + 1)
        est.fit(X[idx])
        selections.append(est.get_support(indices=True))
    pairwise = []
    for i in range(len(selections)):
        for j in range(i + 1, len(selections)):
            pairwise.append(jaccard(selections[i], selections[j]))
    return {
        "mean_jaccard": float(np.mean(pairwise)) if pairwise else 1.0,
        "std_jaccard": float(np.std(pairwise)) if pairwise else 0.0,
        "selections": selections,
    }


def substitution_aware_stability(selector, X, reference_groups, n_repeats=10, sample_fraction=0.8, random_state=0):
    """Subsample stability after mapping selected columns to frozen CGS groups.

    ``reference_groups`` should be created once on a reference dataset by
    ``GSRSubstitutionSelector.get_substitution_groups()``. This function is an
    evaluation diagnostic; it does not alter the selector or its feature scores.
    """
    from .equivalence import substitution_aware_jaccard

    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    selections = []
    for r in range(int(n_repeats)):
        idx = rng.choice(
            X.shape[0],
            size=max(3, int(round(sample_fraction * X.shape[0]))),
            replace=False,
        )
        est = clone(selector)
        if hasattr(est, "random_state"):
            est.set_params(random_state=random_state + r + 1)
        est.fit(X[idx])
        selections.append(est.get_support(indices=True))

    exact = []
    aware = []
    for i in range(len(selections)):
        for j in range(i + 1, len(selections)):
            exact.append(jaccard(selections[i], selections[j]))
            aware.append(
                substitution_aware_jaccard(
                    selections[i], selections[j], reference_groups, X.shape[1]
                )
            )
    return {
        "mean_exact_jaccard": float(np.mean(exact)) if exact else 1.0,
        "mean_substitution_aware_jaccard": float(np.mean(aware)) if aware else 1.0,
        "absolute_gain": float(np.mean(aware) - np.mean(exact)) if exact else 0.0,
        "selections": selections,
    }
