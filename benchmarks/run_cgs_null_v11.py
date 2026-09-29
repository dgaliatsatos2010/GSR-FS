from pathlib import Path
import numpy as np
import pandas as pd
from gsrfs import GSRSubstitutionSelector
from gsrfs.synthetic import null_independent

OUT=Path(__file__).resolve().parent/'results'/'v11'; OUT.mkdir(parents=True,exist_ok=True)
TH=[0.80,0.85,0.90]
rows=[]
for seed in range(50):
    X,_,_,name=null_independent(seed=seed,n=160,p=20)
    est=GSRSubstitutionSelector(
        n_features=3, substitution_threshold=0.0,
        n_permutations=6,max_pairs=2500,support_engine='vectorized',
        residual_engine='batched_cd',random_state=seed,
    ).fit(X)
    sel=set(est.get_support(indices=True).tolist())
    for th in TH:
        assigned=0
        maxsim=[]
        for j in range(X.shape[1]):
            if j in sel: continue
            vals=est.substitution_similarity_[:,j]
            m=float(np.nanmax(vals)); maxsim.append(m)
            assigned += int(m>=th)
        rows.append({'seed':seed,'threshold':th,'assigned_unselected':assigned,
                     'assignment_rate':assigned/(X.shape[1]-len(sel)),
                     'max_unselected_similarity':max(maxsim) if maxsim else np.nan})
raw=pd.DataFrame(rows); raw.to_csv(OUT/'cgs_null_raw.csv',index=False)
summary=raw.groupby('threshold').agg(mean_assignment_rate=('assignment_rate','mean'),
    any_assignment_rate=('assigned_unselected',lambda x: float(np.mean(np.asarray(x)>0))),
    mean_max_similarity=('max_unselected_similarity','mean'),
    q95_max_similarity=('max_unselected_similarity',lambda x: float(np.quantile(x,.95)))).reset_index()
summary.to_csv(OUT/'cgs_null_summary.csv',index=False)
print(summary.to_string(index=False))
