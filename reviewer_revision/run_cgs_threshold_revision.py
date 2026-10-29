import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import combinations
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import GSRSelector,GSRSubstitutionSelector,redundant_blocks,nonlinear_moons,three_factor_blobs
TH=[.75,.80,.85,.90,.95]; REPS=30
GENS=[redundant_blocks,nonlinear_moons,three_factor_blobs]
def jac(a,b):
 A=set(map(int,a));B=set(map(int,b)); return len(A&B)/max(len(A|B),1)
def groups_from_sim(est,t):
 sel=est.selected_indices_.tolist(); sim=est.substitution_similarity_; p=est.n_features_in_; groups={r:[r] for r in sel}
 for j in range(p):
  if j in sel or not sel: continue
  vals=sim[:,j]
  if np.isfinite(vals).any():
   a=int(np.nanargmax(vals))
   if float(vals[a])>=t: groups[sel[a]].append(j)
 return {r:sorted(v) for r,v in groups.items()}
def labels(groups,p):
 lab=np.arange(p)+10_000_000
 for gid,(r,mem) in enumerate(sorted(groups.items())):
  for j in mem: lab[j]=gid
 return lab
def group_jac(a,b,groups,p):
 lab=labels(groups,p); A={int(lab[j]) for j in a}; B={int(lab[j]) for j in b}; return len(A&B)/max(len(A|B),1)
def true_recall(groups,truth,p):
 lab=labels(groups,p); return np.mean([len({int(lab[j]) for j in g})==1 for g in truth])
def noise_rate(groups,truth,p):
 rel={j for g in truth for j in g}; noise=[j for j in range(p) if j not in rel]; grouped={j for v in groups.values() for j in v}; return np.mean([j in grouped for j in noise]) if noise else 0
def one(fn_name,seed):
 fn={'redundant_blocks':redundant_blocks,'nonlinear_moons':nonlinear_moons,'three_factor_blobs':three_factor_blobs}[fn_name]
 X,_,truth,name=fn(seed=seed,n=180,noise_features=12); k=len(truth)
 kw=dict(n_features=k,n_permutations=20,max_pairs=3000,support_engine='vectorized',residual_engine='batched_cd',random_state=seed)
 full=GSRSubstitutionSelector(substitution_threshold=0.0,**kw).fit(X)
 rng=np.random.default_rng(11000+seed); sels=[]
 for r in range(5):
  ix=rng.choice(len(X),size=max(30,int(.8*len(X))),replace=False)
  est=GSRSelector(n_features=k,n_permutations=20,max_pairs=3000,support_engine='vectorized',residual_engine='batched_cd',random_state=seed*100+r+1).fit(X[ix])
  sels.append(est.selected_indices_.tolist())
 exact=np.mean([jac(a,b) for a,b in combinations(sels,2)])
 out=[]
 for t in TH:
  gr=groups_from_sim(full,t); gj=np.mean([group_jac(a,b,gr,X.shape[1]) for a,b in combinations(sels,2)])
  false=0
  for rep,mem in gr.items():
   tg=next((g for g in truth if rep in g),None)
   if tg is None: false+=max(len(mem)-1,0)
   else: false+=len(set(mem)-set(tg)-{rep})
  out.append(dict(scenario=name,seed=seed,threshold=t,exact_jaccard=exact,group_jaccard=gj,stability_gain=gj-exact,
     true_group_recall=true_recall(gr,truth,X.shape[1]),noise_assignment_rate=noise_rate(gr,truth,X.shape[1]),false_assignments=false,
     non_singleton=sum(len(v)>1 for v in gr.values()),n_substitutes=sum(max(0,len(v)-1) for v in gr.values())))
 return out
rows=[]
with ProcessPoolExecutor(max_workers=5) as ex:
 fs={ex.submit(one,fn.__name__,s):(fn.__name__,s) for fn in GENS for s in range(REPS)}
 for i,f in enumerate(as_completed(fs),1): rows.extend(f.result());
raw=pd.DataFrame(rows).sort_values(['scenario','seed','threshold']);raw.to_csv('/mnt/data/cgs_threshold_revision_raw.csv',index=False)
sumdf=raw.groupby('threshold',as_index=False).agg(runs=('seed','count'),exact_jaccard=('exact_jaccard','mean'),group_jaccard=('group_jaccard','mean'),stability_gain=('stability_gain','mean'),true_group_recall=('true_group_recall','mean'),noise_assignment_rate=('noise_assignment_rate','mean'),total_false_assignments=('false_assignments','sum'),runs_with_false=('false_assignments',lambda x:int((x>0).sum())),mean_non_singleton=('non_singleton','mean'),mean_substitutes=('n_substitutes','mean'))
sumdf.to_csv('/mnt/data/cgs_threshold_revision_summary.csv',index=False)
by=raw.groupby(['scenario','threshold'],as_index=False).agg(group_jaccard=('group_jaccard','mean'),stability_gain=('stability_gain','mean'),true_group_recall=('true_group_recall','mean'),noise_assignment_rate=('noise_assignment_rate','mean'),false_assignments=('false_assignments','sum'))
by.to_csv('/mnt/data/cgs_threshold_revision_by_scenario.csv',index=False)
print(sumdf.to_string(index=False)); print('\nBY SCENARIO\n',by.to_string(index=False))
