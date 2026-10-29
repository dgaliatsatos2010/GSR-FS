import os,sys,time,threading,psutil
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
import numpy as np,pandas as pd
from gsrfs_revision_core import GSRSelector
p=int(sys.argv[1]); rep=int(sys.argv[2]); N=250;B=25;M=2000
seed=940000+p+rep;rng=np.random.default_rng(seed);X=rng.normal(size=(N,p))
if p>=6:
 X[:,2]=X[:,0]+.05*rng.normal(size=N);X[:,3]=X[:,1]+.05*rng.normal(size=N);X[:,5]=.7*X[:,4]+.3*rng.normal(size=N)
proc=psutil.Process();base=proc.memory_info().rss;peak=[base];stop=[False]
def mon():
 while not stop[0]:
  try: peak[0]=max(peak[0],proc.memory_info().rss)
  except: pass
  time.sleep(.003)
th=threading.Thread(target=mon);th.start();t=time.perf_counter();e=GSRSelector(n_features=8,n_permutations=B,max_pairs=M,pair_sampler='balanced',support_engine='vectorized',residual_engine='auto',random_state=seed).fit(X);sec=time.perf_counter()-t;stop[0]=True;th.join()
row=dict(p=p,rep=rep,n=N,B=B,max_pairs=M,n_pairs_used=int(e.n_pairs_used_),fit_seconds=sec,peak_rss_mb=peak[0]/2**20,baseline_rss_mb=base/2**20,incremental_peak_mb=(peak[0]-base)/2**20,selected_k=int(e.n_features_selected_),completed=1)
pd.DataFrame([row]).to_csv(f'r2_highdim_p{p}_rep{rep}.csv',index=False);print(row)
