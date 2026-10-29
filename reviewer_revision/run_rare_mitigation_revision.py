import sys,time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import GSRSelector,outlier_contamination,group_coverage,relevant_precision,redundancy_rate
ALPHAS=[.05,.025,.01]; BETAS=[2.0,3.0,4.0]; REPS=50

def run_cfg(cfg):
    a,b=cfg; rows=[]
    for seed in range(REPS):
        X,y,groups,name=outlier_contamination(seed=seed,n=240,noise_features=20,outlier_fraction=.08)
        t=time.perf_counter()
        est=GSRSelector(n_features='auto',n_permutations=100,alpha=a,max_pairs=5000,pair_sampler='balanced',robust=True,
                        coverage_power=b,residual_power=.5,allow_empty=True,support_engine='vectorized',residual_engine='auto',random_state=seed).fit(X)
        idx=est.get_support(indices=True)
        rows.append(dict(alpha=a,coverage_power=b,seed=seed,n_selected=len(idx),group_coverage=group_coverage(idx,groups),
                         relevant_precision=relevant_precision(idx,groups),redundancy_rate=redundancy_rate(idx,groups),
                         fit_seconds=time.perf_counter()-t))
    return rows
rows=[]
with ProcessPoolExecutor(max_workers=5) as ex:
    futs={ex.submit(run_cfg,c):c for c in [(a,b) for a in ALPHAS for b in BETAS]}
    for f in as_completed(futs): rows.extend(f.result()); print('finished',futs[f],flush=True)
raw=pd.DataFrame(rows).sort_values(['alpha','coverage_power','seed'])
raw.to_csv('/mnt/data/rare_geometry_mitigation_raw.csv',index=False)
summary=raw.groupby(['alpha','coverage_power'],as_index=False).agg(runs=('seed','count'),mean_k=('n_selected','mean'),sd_k=('n_selected','std'),median_k=('n_selected','median'),
    mean_group_coverage=('group_coverage','mean'),mean_precision=('relevant_precision','mean'),mean_redundancy=('redundancy_rate','mean'),mean_fit_seconds=('fit_seconds','mean'))
summary.to_csv('/mnt/data/rare_geometry_mitigation_summary.csv',index=False)
print(summary.to_string(index=False))
# original 15-run exact benchmark for validation
orig=raw[(raw.alpha==.05)&(raw.coverage_power==2.0)&(raw.seed<15)]
print('\nORIGINAL FIRST15:',orig[['n_selected','group_coverage','relevant_precision','redundancy_rate']].mean().to_dict())
