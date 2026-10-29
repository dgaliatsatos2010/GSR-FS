import sys,time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import GSRSelector

def make(seed,n=300,n_noise=20,nuisance=False):
    rng=np.random.default_rng(seed)
    y=rng.integers(0,2,size=n)
    mu=np.where(y==0,-1.5,1.5)
    z1=rng.normal(mu,.70); z2=rng.normal(mu*.75,.80)
    cols=[z1,z2,z1+rng.normal(0,.08,n),z2+rng.normal(0,.08,n)]
    signal=set(range(4)); nuisance_idx=set()
    if nuisance:
        u=rng.normal(size=n)
        # three extremely coherent nuisance copies, independent of y
        start=len(cols)
        cols += [u, u+rng.normal(0,.015,n), u+rng.normal(0,.020,n)]
        nuisance_idx=set(range(start,start+3))
    cols += [rng.normal(size=n) for _ in range(n_noise)]
    X=np.column_stack(cols)
    return X,y,signal,nuisance_idx

def one(seed,nuisance,mode):
    X,y,signal,nuis=make(seed,nuisance=nuisance)
    kwargs=dict(n_permutations=100,max_pairs=3000,pair_sampler='balanced',coverage_power=2.0,use_residual=True,residual_power=.5,
                support_engine='vectorized',residual_engine='auto',random_state=seed)
    if mode=='fixed2': kwargs['n_features']=2
    else: kwargs.update(n_features='auto',alpha=.05,allow_empty=True)
    est=GSRSelector(**kwargs).fit(X)
    s=set(map(int,est.get_support(indices=True)))
    # feature-level top categories and scores
    top=int(np.argmax(est.effective_support_))
    return dict(scenario='signal_plus_coherent_nuisance' if nuisance else 'clean_signal',seed=seed,mode=mode,
        n_selected=len(s),signal_selected=len(s&signal),nuisance_selected=len(s&nuis),noise_selected=len(s-signal-nuis),
        signal_precision=(len(s&signal)/len(s) if s else 0.0),signal_recall=len(s&signal)/len(signal),
        any_nuisance=int(bool(s&nuis)),top_feature=top,top_is_signal=int(top in signal),top_is_nuisance=int(top in nuis),
        mean_signal_support=float(np.mean(est.effective_support_[list(signal)])),
        mean_nuisance_support=float(np.mean(est.effective_support_[list(nuis)])) if nuis else np.nan,
        mean_noise_support=float(np.mean(est.effective_support_[list(set(range(X.shape[1]))-signal-nuis)])),
        selected=','.join(map(str,sorted(s))))

tasks=[(seed,nuis,mode) for nuis in [False,True] for mode in ['fixed2','auto'] for seed in range(100)]
rows=[]
with ProcessPoolExecutor(max_workers=5) as ex:
    futs=[ex.submit(one,*t) for t in tasks]
    for i,f in enumerate(as_completed(futs),1):
        rows.append(f.result())
        if i%100==0: print('done',i,flush=True)
raw=pd.DataFrame(rows).sort_values(['scenario','mode','seed'])
raw.to_csv('/mnt/data/informative_nuisance_revision_raw.csv',index=False)
summary=raw.groupby(['scenario','mode'],as_index=False).agg(
    runs=('seed','count'),mean_k=('n_selected','mean'),signal_precision=('signal_precision','mean'),signal_recall=('signal_recall','mean'),
    mean_signal_selected=('signal_selected','mean'),mean_nuisance_selected=('nuisance_selected','mean'),nuisance_any_rate=('any_nuisance','mean'),
    top_signal_rate=('top_is_signal','mean'),top_nuisance_rate=('top_is_nuisance','mean'),
    mean_signal_support=('mean_signal_support','mean'),mean_nuisance_support=('mean_nuisance_support','mean'),mean_noise_support=('mean_noise_support','mean'))
summary.to_csv('/mnt/data/informative_nuisance_revision_summary.csv',index=False)
print(summary.to_string(index=False))
