from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from gsrfs import GSRSelector, GSRConcordanceSelector
from gsrfs.synthetic import redundant_blocks, three_factor_blobs, nonlinear_moons

OUT = Path(__file__).resolve().parent / 'results' / 'v10'
OUT.mkdir(parents=True, exist_ok=True)


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b) if (a | b) else 1.0


def stability(factory, X, seed, repeats=4, frac=0.8):
    rng = np.random.default_rng(5000 + int(seed))
    sels = []
    for r in range(repeats):
        idx = rng.choice(len(X), size=int(round(frac * len(X))), replace=False)
        est = factory(int(seed) * 100 + r)
        est.fit(X[idx])
        sels.append(est.get_support(indices=True))
    vals = [jaccard(sels[i], sels[j]) for i, j in combinations(range(len(sels)), 2)]
    return float(np.mean(vals)), sels


def main():
    rows = []
    for fn in [redundant_blocks, three_factor_blobs, nonlinear_moons]:
        for seed in range(5):
            X, _, groups, name = fn(seed=seed, n=180, noise_features=12)
            k = max(2, len(groups))
            for method in ['GSR-base', 'GSR-JGC']:
                if method == 'GSR-base':
                    factory = lambda rs: GSRSelector(
                        n_features=k, n_permutations=6, max_pairs=3000,
                        support_engine='vectorized', random_state=rs,
                    )
                else:
                    factory = lambda rs: GSRConcordanceSelector(
                        n_features=k, tie_tolerance=0.10, concordance_weight=0.0,
                        n_blocks=4, n_permutations=6, max_pairs=3000,
                        support_engine='vectorized', random_state=rs,
                    )
                est = factory(seed)
                est.fit(X)
                selected = est.get_support(indices=True)
                coverage = float(np.mean([any(j in g for j in selected) for g in groups]))
                stab, _ = stability(factory, X, seed)
                rows.append({
                    'scenario': name,
                    'seed': seed,
                    'method': method,
                    'k': k,
                    'selected': ','.join(map(str, selected.tolist())),
                    'latent_group_coverage': coverage,
                    'mean_subsample_jaccard': stab,
                })
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / 'jgc_synthetic_confirmation_raw.csv', index=False)
    summary = raw.groupby('method')[['latent_group_coverage','mean_subsample_jaccard']].agg(['mean','std'])
    summary.to_csv(OUT / 'jgc_synthetic_confirmation_summary.csv')
    by = raw.groupby(['scenario','method'])[['latent_group_coverage','mean_subsample_jaccard']].mean().reset_index()
    by.to_csv(OUT / 'jgc_synthetic_confirmation_by_scenario.csv', index=False)
    print(summary)

if __name__ == '__main__':
    main()
