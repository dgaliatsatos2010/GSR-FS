import os,time
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from gsrfs_revision_core import GSRSelector,outlier_contamination,group_coverage,relevant_precision,redundancy_rate
CLIPS=[3.0,4.0,4.5,5.0,6.0];REPS=30
def run_clip(c):
 rows=[]
 for seed in range(REPS):
  X,_,groups,_=outlier_contamination(seed=960000+seed,n=240,noise_features=20,outlier_fraction=.08)
  t=time.perf_counter();e=GSRSelector(n_features='auto',n_permutations=100,alpha=.05,max_pairs=5000,pair_sampler='balanced',robust=True,clip_z=c,coverage_power=2.,residual_power=.5,allow_empty=True,support_engine='vectorized',residual_engine='auto',random_state=960000+seed).fit(X);idx=e.selected_indices_
  rows.append(dict(clip_z=c,seed=seed,n_selected=len(idx),group_coverage=group_coverage(idx,groups),relevant_precision=relevant_precision(idx,groups),redundancy=redundancy_rate(idx,groups),fit_seconds=time.perf_counter()-t))
 return rows
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=5) as ex: nested=list(ex.map(run_clip,CLIPS))
 d=pd.DataFrame([x for sub in nested for x in sub]);d.to_csv('r2_clip_sensitivity_raw.csv',index=False)
 s=d.groupby('clip_z',as_index=False).agg(runs=('seed','size'),mean_k=('n_selected','mean'),sd_k=('n_selected','std'),mean_group_coverage=('group_coverage','mean'),mean_precision=('relevant_precision','mean'),mean_redundancy=('redundancy','mean'),mean_fit_seconds=('fit_seconds','mean'))
 s.to_csv('r2_clip_sensitivity_summary.csv',index=False);print(s.to_string(index=False))
