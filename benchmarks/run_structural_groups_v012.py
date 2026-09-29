from pathlib import Path
import json, math, time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_diabetes

from gsrfs import GSRSelector, GSRStructuralGroupSelector, substitution_aware_stability

OUT = Path(__file__).resolve().parent / 'results' / 'v12'
OUT.mkdir(parents=True, exist_ok=True)


def datasets():
    longley = sm.datasets.longley.load_pandas().exog.copy()
    macro = sm.datasets.macrodata.load_pandas().data.copy().drop(columns=['year','quarter'])
    diabetes = pd.DataFrame(load_diabetes().data, columns=load_diabetes().feature_names)
    return {
        'Longley_exog': longley,
        'Macrodata_numeric': macro,
        'Diabetes_features': diabetes,
    }

rows=[]
group_rows=[]
for di, (name, df) in enumerate(datasets().items()):
    X = df.to_numpy(dtype=float)
    p = X.shape[1]
    k = min(p-1, max(2, int(math.ceil(math.sqrt(p)))))
    common = dict(
        n_features=k,
        n_permutations=40,
        max_pairs=15000,
        pair_sampler='balanced',
        support_engine='vectorized',
        residual_engine='batched_cd',
        residual_power=0.5,
        random_state=710 + di,
    )
    t0=time.perf_counter()
    est = GSRStructuralGroupSelector(substitution_threshold=0.85, **common).fit(df)
    fit_s=time.perf_counter()-t0
    groups = est.get_structural_groups()
    base = GSRSelector(**common)
    stab = substitution_aware_stability(
        base, X, est.get_substitution_groups(),
        n_repeats=12, sample_fraction=0.8, random_state=1200+di,
    )
    summary = est.get_group_summary()
    rows.append({
        'dataset': name,
        'n_samples': X.shape[0],
        'n_features': p,
        'k': k,
        'selected_representatives': '|'.join(est.get_representatives(names=True)),
        'n_structural_groups': summary['n_structural_groups'],
        'n_non_singleton_groups': summary['n_non_singleton_groups'],
        'n_contextual_substitutes': summary['n_contextual_substitutes'],
        'mean_exact_jaccard': stab['mean_exact_jaccard'],
        'mean_group_aware_jaccard': stab['mean_substitution_aware_jaccard'],
        'absolute_stability_gain': stab['absolute_gain'],
        'fit_seconds': fit_s,
    })
    for g in groups:
        group_rows.append({
            'dataset': name,
            'group_id': g['group_id'],
            'representative': g['representative'],
            'members': '|'.join(g['members']),
            'n_members': g['n_members'],
            'n_substitutes': g['n_substitutes'],
            'mean_member_cgs_similarity': g['mean_member_cgs_similarity'],
            'min_member_cgs_similarity': g['min_member_cgs_similarity'],
            'representative_selection_score': g['representative_selection_score'],
            'representative_effective_support': g['representative_effective_support'],
        })

summary_df=pd.DataFrame(rows)
groups_df=pd.DataFrame(group_rows)
summary_df.to_csv(OUT/'structural_group_fresh_summary.csv', index=False)
groups_df.to_csv(OUT/'structural_group_fresh_groups.csv', index=False)
print(summary_df.to_string(index=False))
print('\nGROUPS\n', groups_df.to_string(index=False))
