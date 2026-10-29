import sys,time,resource,argparse,os
import numpy as np
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import GSRSelector
ap=argparse.ArgumentParser();ap.add_argument('--n',type=int);ap.add_argument('--p',type=int);ap.add_argument('--M',type=int);ap.add_argument('--B',type=int);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args()
rng=np.random.default_rng(a.seed);X=rng.normal(size=(a.n,a.p))
if a.p>=6:
 X[:,2]=X[:,0]+.05*rng.normal(size=a.n);X[:,3]=X[:,1]+.05*rng.normal(size=a.n);X[:,5]=.7*X[:,4]+.3*rng.normal(size=a.n)
t=time.perf_counter();est=GSRSelector(n_features=min(8,a.p),n_permutations=a.B,max_pairs=a.M,support_engine='vectorized',residual_engine='auto',random_state=a.seed).fit(X);dt=time.perf_counter()-t
rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0
print(f'{dt},{rss},{est.n_pairs_used_}')
