from pathlib import Path
import numpy as np, pandas as pd
from gsrfs import GSRSelector
from gsrfs.partition import persistent_balanced_gap_evidence, _average_rank01
OUT=Path(__file__).resolve().parent/'results'/'v09'; OUT.mkdir(parents=True,exist_ok=True)

def scenario(seed,kind,n=180,noise=12):
 r=np.random.default_rng(seed)
 if kind=='partition_vs_continuum':
  b=r.integers(0,2,n); part=np.where(b==0,-2.2,2.2)+r.normal(0,.35,n); u=r.normal(size=n)
  X=np.column_stack([part,*[u+r.normal(0,.08,n) for _ in range(4)],r.normal(size=(n,noise))]); groups=[[0],[1,2,3,4]]; k=2
 elif kind=='partition_redundant':
  b=r.integers(0,2,n); p1=np.where(b==0,-2,2)+r.normal(0,.4,n); p2=p1+r.normal(0,.06,n); u=r.normal(size=n)
  X=np.column_stack([p1,p2,u,u+r.normal(0,.08,n),r.normal(size=(n,noise))]); groups=[[0,1],[2,3]]; k=2
 elif kind=='smooth_structure':
  u=r.normal(size=n); X=np.column_stack([u,u+r.normal(0,.07,n),.7*u+r.normal(0,.15,n),r.normal(size=(n,noise))]); groups=[[0,1,2]]; k=1
 elif kind=='three_partitions':
  b=r.integers(0,3,n); p=np.array([-2.5,0,2.5])[b]+r.normal(0,.28,n); u=r.normal(size=n)
  X=np.column_stack([p,u,u+r.normal(0,.08,n),r.normal(size=(n,noise))]); groups=[[0],[1,2]]; k=2
 return X,groups,k

def cov(sel,groups):
 s=set(sel); return np.mean([bool(s.intersection(g)) for g in groups])
def red(sel,groups):
 s=set(sel); return sum(max(0,len(s.intersection(g))-1) for g in groups)/max(len(sel),1)
def greedy(base,X,k,w):
 Xs=(X-base.location_)/base.scale_
 if base.robust and base.clip_z is not None: Xs=np.clip(Xs,-base.clip_z,base.clip_z)
 pe=np.array([persistent_balanced_gap_evidence(Xs[:,j]) for j in range(X.shape[1])])
 fused=(1-w)*_average_rank01(base.effective_support_)+w*_average_rank01(pe)
 ss=np.random.SeedSequence(base.random_state); pair_seed,_=ss.spawn(2); rng=np.random.default_rng(pair_seed)
 pi,pj=base._sample_pairs(X.shape[0],rng); G=base._geometry_profiles(Xs,pi,pj); H=G/(np.linalg.norm(G,axis=0,keepdims=True)+base.eps)
 selected=[]
 for _ in range(k):
  rem=[j for j in range(X.shape[1]) if j not in selected]; rr=base._candidate_residuals(H,selected,rem)
  sc=fused[rem]*np.power(rr,base.residual_power); selected.append(rem[int(np.argmax(sc))])
 return selected
rows=[]; scens=['partition_vs_continuum','partition_redundant','smooth_structure','three_partitions']
for seed in range(6):
 for kind in scens:
  X,groups,k=scenario(seed,kind)
  base=GSRSelector(n_features=k,n_permutations=10,max_pairs=3000,residual_power=.5,random_state=seed).fit(X)
  rows.append([seed,kind,'GSR-base',0,cov(base.selected_indices_,groups),red(base.selected_indices_,groups)])
  for w in [.20,.35,.50,.70]:
   sel=greedy(base,X,k,w); rows.append([seed,kind,'GSR-PBGE',w,cov(sel,groups),red(sel,groups)])
df=pd.DataFrame(rows,columns=['seed','scenario','method','weight','coverage','redundancy'])
df.to_csv(OUT/'partition_development_raw.csv',index=False)
sm=df.groupby(['method','weight'])[['coverage','redundancy']].mean().reset_index(); sm.to_csv(OUT/'partition_development_summary.csv',index=False)
by=df.groupby(['scenario','method','weight'])[['coverage','redundancy']].mean().reset_index(); by.to_csv(OUT/'partition_development_by_scenario.csv',index=False)
print(sm.to_string(index=False)); print('\n',by.to_string(index=False))
