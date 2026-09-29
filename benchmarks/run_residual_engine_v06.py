from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from gsrfs import GSRSelector

OUT=Path(__file__).resolve().parent/'results'/'v06'; OUT.mkdir(parents=True,exist_ok=True)
X=load_breast_cancer().data
rows=[]
for m in [10000,20000,50000]:
    fitted={}
    for eng in ['nnls','batched_cd']:
        t=time.perf_counter()
        s=GSRSelector(n_features=8,n_permutations=10,max_pairs=m,pair_sampler='balanced',random_state=0,support_engine='vectorized',residual_engine=eng).fit(X)
        dt=time.perf_counter()-t
        fitted[eng]=s
        rows.append({'max_pairs':m,'engine':eng,'fit_seconds':dt,'selected':','.join(map(str,s.get_support(indices=True).tolist()))})
    diff=float(np.max(np.abs(fitted['nnls'].feature_scores_-fitted['batched_cd'].feature_scores_)))
    same=bool(np.array_equal(fitted['nnls'].get_support(indices=True),fitted['batched_cd'].get_support(indices=True)))
    for row in rows[-2:]:
        row['same_selection_path']=same
        row['max_abs_score_diff']=diff

df=pd.DataFrame(rows); df.to_csv(OUT/'residual_engine_v06_raw.csv',index=False)
piv=df.pivot(index='max_pairs',columns='engine',values='fit_seconds').reset_index()
piv['speedup_batched_vs_nnls']=piv['nnls']/piv['batched_cd']
piv.to_csv(OUT/'residual_engine_v06_summary.csv',index=False)
print(piv.to_string(index=False))
