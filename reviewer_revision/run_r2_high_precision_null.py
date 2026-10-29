import os
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('OPENBLAS_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1'); os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import beta as beta_dist
from gsrfs_revision_core import GSRSelector, null_independent
N=240; P=24; REPS=1000; B=200; M=5000; ALPHAS=[0.01,0.025,0.05,0.10]
def qhigher(x,q):
    try:return float(np.quantile(x,q,method='higher'))
    except TypeError:return float(np.quantile(x,q,interpolation='higher'))
def ci(k,n):
    a=.025; return (0.0 if k==0 else float(beta_dist.ppf(a,k,n-k+1)),1.0 if k==n else float(beta_dist.ppf(1-a,k+1,n-k)))
def one(rep):
    seed=920000+rep; X,_,_,_=null_independent(seed=seed,n=N,p=P,dist='gaussian')
    est=GSRSelector(n_features='auto',n_permutations=B,alpha=.05,max_pairs=M,pair_sampler='balanced',support_engine='vectorized',residual_engine='auto',coverage_power=2.0,clip_z=4.5,allow_empty=True,random_state=seed).fit(X)
    obs=float(np.max(est.effective_support_)); mx=np.asarray(est.max_null_distribution_,float)
    row={'rep':rep,'seed':seed,'n':N,'p':P,'B':B,'max_pairs':M,'max_observed_effective_support':obs,'selected_k_alpha_0.05':int(est.n_features_selected_)}
    for a in ALPHAS:
        thr=qhigher(mx,1-a); row[f'threshold_alpha_{a:g}']=thr; row[f'nonempty_alpha_{a:g}']=int(obs>=thr)
    return row
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=4) as ex: rows=list(ex.map(one,range(REPS),chunksize=5))
    raw=pd.DataFrame(rows); raw.to_csv('r2_high_precision_null_raw.csv',index=False)
    out=[]
    for a in ALPHAS:
        k=int(raw[f'nonempty_alpha_{a:g}'].sum()); lo,hi=ci(k,REPS)
        out.append({'distribution':'independent_gaussian','n':N,'p':P,'replicates':REPS,'B':B,'max_pairs':M,'alpha':a,'nonempty_count':k,'nonempty_rate':k/REPS,'ci95_low':lo,'ci95_high':hi,'mc_p_resolution':1/(B+1)})
    s=pd.DataFrame(out); s.to_csv('r2_high_precision_null_summary.csv',index=False); print(s.to_string(index=False))
