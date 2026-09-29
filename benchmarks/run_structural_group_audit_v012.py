from pathlib import Path
import numpy as np
import pandas as pd
from gsrfs import GSRStructuralGroupSelector
from gsrfs.synthetic import redundant_blocks

OUT=Path(__file__).resolve().parent/'results'/'v12'; OUT.mkdir(parents=True, exist_ok=True)

rows=[]
# Null panel
for seed in range(30):
    rng=np.random.default_rng(9000+seed)
    X=rng.normal(size=(140,12))
    est=GSRStructuralGroupSelector(
        n_features=4, n_permutations=10, max_pairs=5000,
        support_engine='vectorized', residual_engine='batched_cd',
        substitution_threshold=0.85, random_state=seed,
    ).fit(X)
    sm=est.get_group_summary()
    rows.append({'regime':'null','seed':seed,'known_group_recovery':np.nan,
                 'false_substitute_assignments':sm['n_contextual_substitutes'],
                 'n_non_singleton_groups':sm['n_non_singleton_groups']})

# Known redundant panel
for seed in range(20):
    X,_,truth,_=redundant_blocks(seed=100+seed,n=180,noise_features=8)
    est=GSRStructuralGroupSelector(
        n_features=2, n_permutations=10, max_pairs=5000,
        support_engine='vectorized', residual_engine='batched_cd',
        substitution_threshold=0.85, random_state=100+seed,
    ).fit(X)
    groups=est.get_substitution_groups()
    recovered=[]
    for rep,members in groups.items():
        tg=next((g for g in truth if rep in g),None)
        if tg is not None:
            recovered.append(float(set(tg).issubset(set(members))))
    # Any assigned unselected member outside the true group of its representative is false.
    false=0
    for rep,members in groups.items():
        tg=next((g for g in truth if rep in g),None)
        if tg is None:
            false += max(len(members)-1,0)
        else:
            false += len(set(members)-set(tg)-{rep})
    sm=est.get_group_summary()
    rows.append({'regime':'redundant_blocks','seed':seed,
                 'known_group_recovery':float(np.mean(recovered)) if recovered else np.nan,
                 'false_substitute_assignments':false,
                 'n_non_singleton_groups':sm['n_non_singleton_groups']})

df=pd.DataFrame(rows); df.to_csv(OUT/'structural_group_audit_raw.csv',index=False)
summary=df.groupby('regime',as_index=False).agg(
    runs=('seed','count'),
    mean_known_group_recovery=('known_group_recovery','mean'),
    total_false_substitute_assignments=('false_substitute_assignments','sum'),
    runs_with_false_assignments=('false_substitute_assignments',lambda x:int((x>0).sum())),
    mean_non_singleton_groups=('n_non_singleton_groups','mean'),
)
summary.to_csv(OUT/'structural_group_audit_summary.csv',index=False)
print(summary.to_string(index=False))
