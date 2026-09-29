"""Validate sampled pair schemes against exact all-pairs GSR geometry."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gsrfs import GSRSelector
from gsrfs.metrics import jaccard
from gsrfs.realdata import load_builtin_benchmarks


def main():
    out = Path(__file__).resolve().parent / "results" / "v06"
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    budgets=[5000,10000,20000,50000]
    B=20
    seed=0
    for ds in load_builtin_benchmarks(include_digits=False):
        X=ds.X; p=X.shape[1]
        k=min(max(2,int(round(.25*p))),min(10,p))
        total=X.shape[0]*(X.shape[0]-1)//2
        exact_cap=total+1
        common=dict(n_features=k,n_permutations=B,random_state=seed,
                    support_engine='vectorized',residual_engine='auto')
        t=time.perf_counter()
        exact=GSRSelector(max_pairs=exact_cap,pair_sampler='balanced',**common).fit(X)
        exact_t=time.perf_counter()-t
        exact_sel=exact.get_support(indices=True)
        exact_rank=exact.ranking_.copy()
        for scheme in ['balanced','random']:
            for budget in budgets:
                if budget>=total:
                    sel=exact; dt=exact_t
                else:
                    t=time.perf_counter()
                    sel=GSRSelector(max_pairs=budget,pair_sampler=scheme,**common).fit(X)
                    dt=time.perf_counter()-t
                chosen=sel.get_support(indices=True)
                rows.append({
                    'dataset':ds.name,'scheme':scheme,'max_pairs':budget,
                    'total_pairs':total,'k':k,
                    'jaccard_vs_exact':jaccard(exact_sel,chosen),
                    'rank_spearman_vs_exact':float(spearmanr(exact_rank,sel.ranking_).statistic),
                    'fit_seconds':dt,'exact_seconds':exact_t,
                    'selected':','.join(map(str,chosen.tolist())),
                    'exact_selected':','.join(map(str,exact_sel.tolist())),
                })
    df=pd.DataFrame(rows)
    df.to_csv(out/'pair_sampler_validation_v06_raw.csv',index=False)
    sm=(df.groupby(['scheme','max_pairs'])[['jaccard_vs_exact','rank_spearman_vs_exact','fit_seconds']]
        .mean().reset_index())
    sm.to_csv(out/'pair_sampler_validation_v06_summary.csv',index=False)
    print(sm.to_string(index=False))

if __name__=='__main__':
    main()
