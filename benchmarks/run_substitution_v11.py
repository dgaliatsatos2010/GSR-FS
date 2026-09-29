from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from gsrfs import GSRSelector, GSRSubstitutionSelector, substitution_aware_jaccard
from gsrfs.equivalence import reference_group_labels
from gsrfs.metrics import jaccard
from gsrfs.synthetic import redundant_blocks, three_factor_blobs, nonlinear_moons

OUT = Path(__file__).resolve().parent / 'results' / 'v11'
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [0.80, 0.85, 0.90, 0.95, 0.98]


def base_kwargs(seed, k):
    return dict(
        n_features=k,
        n_permutations=8,
        max_pairs=3000,
        support_engine='vectorized',
        residual_engine='batched_cd',
        random_state=int(seed),
    )


def groups_from_similarity(est, threshold):
    selected = est.get_support(indices=True).tolist()
    sim = est.substitution_similarity_
    p = est.n_features_in_
    groups = {int(r): [int(r)] for r in selected}
    for j in range(p):
        if j in selected:
            continue
        vals = sim[:, j]
        if not np.isfinite(vals).any():
            continue
        a = int(np.nanargmax(vals))
        if float(vals[a]) >= float(threshold):
            groups[int(selected[a])].append(int(j))
    for r in groups:
        groups[r] = sorted(groups[r])
    return groups


def resample_selections(X, seed, k, repeats=5, frac=0.8):
    rng = np.random.default_rng(11000 + int(seed))
    sels = []
    for r in range(repeats):
        idx = rng.choice(len(X), size=max(30, int(round(frac * len(X)))), replace=False)
        est = GSRSelector(**base_kwargs(seed * 100 + r + 1, k)).fit(X[idx])
        sels.append(est.get_support(indices=True).tolist())
    return sels


def mean_pairwise_exact(sels):
    vals = [jaccard(sels[i], sels[j]) for i, j in combinations(range(len(sels)), 2)]
    return float(np.mean(vals)) if vals else 1.0


def mean_pairwise_group(sels, groups, p):
    vals = [substitution_aware_jaccard(sels[i], sels[j], groups, p) for i, j in combinations(range(len(sels)), 2)]
    return float(np.mean(vals)) if vals else 1.0


def true_group_recall(groups, true_groups, p):
    labels = reference_group_labels(groups, p)
    ok = []
    for g in true_groups:
        labs = {int(labels[int(j)]) for j in g}
        ok.append(len(labs) == 1)
    return float(np.mean(ok)) if ok else np.nan


def noise_assignment_rate(groups, true_groups, p):
    relevant = {int(j) for g in true_groups for j in g}
    noise = [j for j in range(p) if j not in relevant]
    if not noise:
        return 0.0
    grouped_members = {j for members in groups.values() for j in members}
    return float(np.mean([j in grouped_members for j in noise]))


def main():
    rows = []
    generators = [redundant_blocks, three_factor_blobs, nonlinear_moons]
    for fn in generators:
        for seed in range(5):
            X, _, true_groups, name = fn(seed=seed, n=180, noise_features=12)
            k = len(true_groups)
            full = GSRSubstitutionSelector(
                substitution_threshold=0.0,
                **base_kwargs(seed, k),
            ).fit(X)
            sels = resample_selections(X, seed, k)
            exact = mean_pairwise_exact(sels)
            for threshold in THRESHOLDS:
                groups = groups_from_similarity(full, threshold)
                rows.append({
                    'scenario': name,
                    'seed': seed,
                    'threshold': threshold,
                    'k': k,
                    'selected_full': ','.join(map(str, full.get_support(indices=True).tolist())),
                    'n_discovered_group_members': sum(len(v) for v in groups.values()),
                    'exact_jaccard': exact,
                    'substitution_aware_jaccard': mean_pairwise_group(sels, groups, X.shape[1]),
                    'true_redundant_group_recall': true_group_recall(groups, true_groups, X.shape[1]),
                    'noise_assignment_rate': noise_assignment_rate(groups, true_groups, X.shape[1]),
                })
    raw = pd.DataFrame(rows)
    raw['stability_gain'] = raw['substitution_aware_jaccard'] - raw['exact_jaccard']
    raw.to_csv(OUT / 'cgs_threshold_sweep_raw.csv', index=False)
    summary = raw.groupby('threshold').agg(
        exact_jaccard=('exact_jaccard','mean'),
        substitution_aware_jaccard=('substitution_aware_jaccard','mean'),
        stability_gain=('stability_gain','mean'),
        true_redundant_group_recall=('true_redundant_group_recall','mean'),
        noise_assignment_rate=('noise_assignment_rate','mean'),
    ).reset_index()
    summary.to_csv(OUT / 'cgs_threshold_sweep_summary.csv', index=False)
    by = raw.groupby(['scenario','threshold']).agg(
        exact_jaccard=('exact_jaccard','mean'),
        substitution_aware_jaccard=('substitution_aware_jaccard','mean'),
        true_redundant_group_recall=('true_redundant_group_recall','mean'),
        noise_assignment_rate=('noise_assignment_rate','mean'),
    ).reset_index()
    by.to_csv(OUT / 'cgs_threshold_sweep_by_scenario.csv', index=False)
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
