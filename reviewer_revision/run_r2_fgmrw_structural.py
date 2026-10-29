import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import itertools,time
import numpy as np,pandas as pd
from run_r2_structural_comparison import make,SCENARIOS
from fgmrw_adapter import fgmrw_ufs
from gsrfs_revision_core import group_coverage,relevant_precision,redundancy_rate
K=5

def one(scen,rep):
    cfg=SCENARIOS[scen]; seed=950000+(0 if scen=='moderate_p100' else 100000)+rep
    X,groups=make(seed,cfg)
    mn=X.min(0); mx=X.max(0); X=(X-mn)/np.where(mx-mn>1e-12,mx-mn,1.)
    t=time.perf_counter(); rank,_=fgmrw_ufs(X,alpha=.1,s=.8); sec=time.perf_counter()-t
    sel=np.asarray(rank[:K],int)
    return dict(scenario=scen,rep=rep,method='FGMRW-UFS',n=cfg['n'],p=X.shape[1],latent_groups=cfg['groups'],proxies_per_group=cfg['proxies'],noise_features=cfg['noise'],proxy_sd=cfg['proxy_sd'],alpha_fgmrw=.1,s_fgmrw=.8,group_coverage=group_coverage(sel,groups),relevant_precision=relevant_precision(sel,groups),redundancy=redundancy_rate(sel,groups),selected_k=len(sel),selected=';'.join(map(str,sel.tolist())),fit_seconds=sec,status='ok')

def mean_jacc(sets):
    vals=[]
    for a,b in itertools.combinations(sets,2):
        u=a|b; vals.append(len(a&b)/len(u) if u else 1.)
    return float(np.mean(vals)) if vals else np.nan

def summarize(d,cfg):
    exact=[]; grouped=[]
    for s in d.selected:
        ids=list(map(int,s.split(';'))) if s else []; exact.append(set(ids)); labs=set()
        for j in ids: labs.add(f'g{j//cfg["proxies"]}' if j<cfg['groups']*cfg['proxies'] else f'n{j}')
        grouped.append(labs)
    return pd.DataFrame([dict(method='FGMRW-UFS',replicates=len(d),mean_group_coverage=d.group_coverage.mean(),sd_group_coverage=d.group_coverage.std(),mean_relevant_precision=d.relevant_precision.mean(),sd_relevant_precision=d.relevant_precision.std(),mean_redundancy=d.redundancy.mean(),sd_redundancy=d.redundancy.std(),mean_selected_k=d.selected_k.mean(),exact_column_stability=mean_jacc(exact),group_aware_stability=mean_jacc(grouped),mean_fit_seconds=d.fit_seconds.mean(),median_fit_seconds=d.fit_seconds.median())])

if __name__=='__main__':
    import sys
    scen=sys.argv[1]; cfg=SCENARIOS[scen]
    workers=int(sys.argv[2]) if len(sys.argv)>2 else 6
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows=list(ex.map(partial(one,scen),range(cfg['reps']),chunksize=1))
    d=pd.DataFrame(rows)
    tag='p100_60' if scen=='moderate_p100' else 'p500_40'
    d.to_csv(f'r2_fgmrw_{tag}_raw.csv',index=False)
    s=summarize(d,cfg); s.to_csv(f'r2_fgmrw_{tag}_summary.csv',index=False)
    print(s.to_string(index=False))
