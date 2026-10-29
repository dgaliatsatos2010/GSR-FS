import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from gsrfs_revision_core import GSRSelector,group_coverage,relevant_precision,redundancy_rate
from r2_baselines import MaxVarianceSelector,LaplacianScoreSelector,SPECSelector,MCFSSelector

SCENARIOS={
 'moderate_p100':dict(n=300,groups=5,proxies=4,noise=80,proxy_sd=.60,B=200,M=5000),
 'larger_p500':dict(n=300,groups=5,proxies=10,noise=450,proxy_sd=.35,B=200,M=2000),
}
REPS=100;K=5
MEANS=np.array([[-2.,0.,2.],[2.,-1.5,.5],[-1.5,2.,0.],[0.,2.,-2.],[1.5,-2.,1.]])

def make(seed,cfg):
 rng=np.random.default_rng(seed);n=cfg['n']; y=np.tile(np.arange(3),int(np.ceil(n/3)))[:n];rng.shuffle(y)
 cols=[];groups=[]
 for g in range(cfg['groups']):
  z=MEANS[g,y]+rng.normal(0,.8,size=n); idx=[]
  for q in range(cfg['proxies']):
   idx.append(len(cols)); cols.append(z+rng.normal(0,cfg['proxy_sd'],size=n))
  groups.append(idx)
 for _ in range(cfg['noise']): cols.append(rng.normal(size=n))
 X=np.column_stack(cols)
 # common unsupervised z-standardization so raw variance cannot trivialize recovery
 mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1.);X=(X-mu)/sd
 return X,y,groups

def one(arg):
 scen,rep=arg;cfg=SCENARIOS[scen];seed=950000+(0 if scen=='moderate_p100' else 100000)+rep;X,y,groups=make(seed,cfg)
 methods={
  'GSR-FS':GSRSelector(n_features=K,n_permutations=cfg['B'],alpha=.05,max_pairs=cfg['M'],pair_sampler='balanced',support_engine='vectorized',residual_engine='auto',random_state=seed),
  'MaxVariance':MaxVarianceSelector(K),
  'LaplacianScore':LaplacianScoreSelector(K),
  'SPEC':SPECSelector(K),
  'MCFS-eigengap':MCFSSelector(K,n_clusters='eigengap'),
 }
 rows=[]
 for name,m in methods.items():
  try:
   m.fit(X); sel=np.asarray(m.selected_indices_,int); status='ok'
   rows.append(dict(scenario=scen,rep=rep,method=name,n=cfg['n'],p=X.shape[1],latent_groups=cfg['groups'],proxies_per_group=cfg['proxies'],noise_features=cfg['noise'],proxy_sd=cfg['proxy_sd'],B=cfg['B'] if name=='GSR-FS' else np.nan,max_pairs=cfg['M'] if name=='GSR-FS' else np.nan,group_coverage=group_coverage(sel,groups),relevant_precision=relevant_precision(sel,groups),redundancy=redundancy_rate(sel,groups),selected_k=len(sel),selected=';'.join(map(str,sel.tolist())),status=status))
  except Exception as e:
   rows.append(dict(scenario=scen,rep=rep,method=name,n=cfg['n'],p=X.shape[1],latent_groups=cfg['groups'],proxies_per_group=cfg['proxies'],noise_features=cfg['noise'],proxy_sd=cfg['proxy_sd'],B=cfg['B'] if name=='GSR-FS' else np.nan,max_pairs=cfg['M'] if name=='GSR-FS' else np.nan,group_coverage=np.nan,relevant_precision=np.nan,redundancy=np.nan,selected_k=np.nan,selected='',status='fail:'+str(e)[:100]))
 return rows

def mean_pair_jaccard(strings):
 sets=[set(map(int,s.split(';'))) if isinstance(s,str) and s else set() for s in strings]; vals=[]
 for i in range(len(sets)):
  for j in range(i+1,len(sets)):
   u=sets[i]|sets[j]; vals.append(len(sets[i]&sets[j])/len(u) if u else 1.)
 return float(np.mean(vals)) if vals else np.nan

if __name__=='__main__':
 args=[(s,r) for s in SCENARIOS for r in range(REPS)]
 with ProcessPoolExecutor(max_workers=4) as ex:
  nested=list(ex.map(one,args,chunksize=1))
 rows=[x for sub in nested for x in sub];d=pd.DataFrame(rows);d.to_csv('r2_structural_comparison_raw.csv',index=False)
 ok=d[d.status=='ok'].copy();
 s=ok.groupby(['scenario','method'],as_index=False).agg(replicates=('rep','size'),mean_group_coverage=('group_coverage','mean'),sd_group_coverage=('group_coverage','std'),mean_relevant_precision=('relevant_precision','mean'),sd_relevant_precision=('relevant_precision','std'),mean_redundancy=('redundancy','mean'),sd_redundancy=('redundancy','std'),mean_selected_k=('selected_k','mean'))
 stab=ok.groupby(['scenario','method']).selected.apply(mean_pair_jaccard).rename('cross_rep_jaccard').reset_index();s=s.merge(stab,on=['scenario','method']);s.to_csv('r2_structural_comparison_summary.csv',index=False);print(s.to_string(index=False))
 print('\nFailures:',d[d.status!='ok'][['scenario','method','status']].drop_duplicates().to_string(index=False))
