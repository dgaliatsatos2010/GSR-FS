import sys
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import *
NAMES=['iris','wine','breast_cancer','anes96','autos_symbol','midwest_state','ccard_ownrent','penguins_species','diamonds_cut','titanic_survived']
TH=[.75,.80,.85,.90,.95]
def jac(a,b):
 A=set(map(int,a));B=set(map(int,b));return len(A&B)/max(len(A|B),1)
def prepare(ds):
 if isinstance(ds.X,pd.DataFrame):
  pre=make_tabular_preprocessor(ds.X); X=pre.fit_transform(ds.X);X=StandardScaler().fit_transform(X)
 else:X=np.asarray(ds.X,float)
 return X
def groups_from_sim(est,t):
 sel=est.selected_indices_.tolist();sim=est.substitution_similarity_;p=est.n_features_in_;gr={r:[r] for r in sel}
 for j in range(p):
  if j in sel or not sel:continue
  vals=sim[:,j]
  if np.isfinite(vals).any():
   a=int(np.nanargmax(vals))
   if float(vals[a])>=t:gr[sel[a]].append(j)
 return gr
def group_jac(a,b,gr,p):
 lab=np.arange(p)+10_000_000
 for gid,(r,mem) in enumerate(sorted(gr.items())):
  for j in mem:lab[j]=gid
 A={int(lab[j]) for j in a};B={int(lab[j]) for j in b};return len(A&B)/max(len(A|B),1)
def work(name):
 ds=next(d for d in all_30_datasets() if d.name==name);X=prepare(ds);p=X.shape[1];k=downstream_k(p)
 full=GSRSubstitutionSelector(n_features=k,n_permutations=12,max_pairs=8000,support_engine='vectorized',residual_engine='auto',random_state=0,substitution_threshold=0).fit(X)
 rng=np.random.default_rng(7700);sels=[]
 for r in range(5):
  ix=rng.choice(len(X),size=max(30,int(.8*len(X))),replace=False)
  est=GSRSelector(n_features=k,n_permutations=12,max_pairs=8000,support_engine='vectorized',residual_engine='auto',random_state=r+1).fit(X[ix]);sels.append(est.selected_indices_.tolist())
 exact=np.mean([jac(a,b) for a,b in combinations(sels,2)]);out=[]
 for t in TH:
  gr=groups_from_sim(full,t);gj=np.mean([group_jac(a,b,gr,p) for a,b in combinations(sels,2)])
  out.append(dict(dataset=name,n=len(X),p=p,k=k,threshold=t,exact_jaccard=exact,group_jaccard=gj,stability_gain=gj-exact,non_singleton=sum(len(v)>1 for v in gr.values()),n_substitutes=sum(max(0,len(v)-1) for v in gr.values())))
 return out
rows=[]
with ProcessPoolExecutor(max_workers=5) as ex:
 fs={ex.submit(work,n):n for n in NAMES}
 for f in as_completed(fs):rows.extend(f.result());print('done',fs[f],flush=True)
raw=pd.DataFrame(rows);raw.to_csv('/mnt/data/cgs_real_threshold_revision_raw.csv',index=False)
s=raw.groupby('threshold',as_index=False).agg(datasets=('dataset','count'),mean_exact=('exact_jaccard','mean'),mean_group=('group_jaccard','mean'),mean_gain=('stability_gain','mean'),datasets_with_gain=('stability_gain',lambda x:int((x>1e-12).sum())),mean_non_singleton=('non_singleton','mean'),mean_substitutes=('n_substitutes','mean'))
s.to_csv('/mnt/data/cgs_real_threshold_revision_summary.csv',index=False)
print(s.to_string(index=False));print('\nRAW\n',raw.to_string(index=False))
