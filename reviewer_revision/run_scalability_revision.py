import sys,time,threading,platform,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,psutil
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import GSRSelector

def make_data(n,p,seed):
 rng=np.random.default_rng(seed); X=rng.normal(size=(n,p))
 if p>=6:
  X[:,2]=X[:,0]+.05*rng.normal(size=n); X[:,3]=X[:,1]+.05*rng.normal(size=n); X[:,5]=.7*X[:,4]+.3*rng.normal(size=n)
 return X

def one(cfg):
 axis,value,n,p,M,B,rep=cfg; seed=4200+rep
 X=make_data(n,p,seed); proc=psutil.Process(); baseline=proc.memory_info().rss; peak=[baseline]; stop=[False]
 def mon():
  while not stop[0]:
   try: peak[0]=max(peak[0],proc.memory_info().rss)
   except: pass
   time.sleep(.005)
 th=threading.Thread(target=mon); th.start(); t=time.perf_counter()
 est=GSRSelector(n_features=min(8,p),n_permutations=B,max_pairs=M,pair_sampler='balanced',support_engine='vectorized',residual_engine='auto',random_state=seed).fit(X)
 elapsed=time.perf_counter()-t; stop[0]=True; th.join()
 return dict(axis=axis,value=value,n=n,p=p,max_pairs=M,n_permutations=B,rep=rep,n_pairs_used=est.n_pairs_used_,fit_seconds=elapsed,
  baseline_mb=baseline/2**20,peak_mb=peak[0]/2**20,incremental_peak_mb=(peak[0]-baseline)/2**20)
configs=[]
for n in [100,250,500,1000,2000]:
 for rep in range(3): configs.append(('n',n,n,50,10000,50,rep))
for p in [20,50,100,200,400]:
 for rep in range(3): configs.append(('p',p,500,p,10000,50,rep))
for M in [1000,2500,5000,10000,20000,50000]:
 for rep in range(3): configs.append(('pair_budget',M,500,50,M,50,rep))
for B in [25,50,100,200]:
 for rep in range(3): configs.append(('permutations',B,500,50,10000,B,rep))
rows=[]
with ProcessPoolExecutor(max_workers=4) as ex:
 fs=[ex.submit(one,c) for c in configs]
 for i,f in enumerate(as_completed(fs),1): rows.append(f.result()); print('done',i,'/',len(fs),flush=True)
raw=pd.DataFrame(rows).sort_values(['axis','value','rep']); raw.to_csv('/mnt/data/scalability_revision_raw.csv',index=False)
s=raw.groupby(['axis','value'],as_index=False).agg(n=('n','first'),p=('p','first'),max_pairs=('max_pairs','first'),n_permutations=('n_permutations','first'),mean_pairs_used=('n_pairs_used','mean'),median_seconds=('fit_seconds','median'),mean_seconds=('fit_seconds','mean'),sd_seconds=('fit_seconds','std'),median_incremental_mb=('incremental_peak_mb','median'),max_incremental_mb=('incremental_peak_mb','max'))
s.to_csv('/mnt/data/scalability_revision_summary.csv',index=False)
info=pd.DataFrame([dict(os=platform.platform(),python=platform.python_version(),cpu=platform.processor() or 'unknown',logical_cores=os.cpu_count(),ram_gb=round(psutil.virtual_memory().total/2**30,2),numpy=np.__version__)])
info.to_csv('/mnt/data/scalability_environment.csv',index=False)
print(s.to_string(index=False)); print('\nENV\n',info.to_string(index=False))
