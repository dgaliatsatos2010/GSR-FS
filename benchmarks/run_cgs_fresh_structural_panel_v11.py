from pathlib import Path
from itertools import combinations
import math
import numpy as np
import pandas as pd
from gsrfs import GSRSelector, GSRSubstitutionSelector, substitution_aware_jaccard
from gsrfs.metrics import jaccard

ROOT=Path('/opt/pyvenv/lib/python3.13/site-packages/plotnine/data')
OUT=Path(__file__).resolve().parent/'results'/'v11'; OUT.mkdir(parents=True,exist_ok=True)

SPECS={
 'penguins': ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','year'],
 'mpg': ['displ','year','cyl','cty','hwy'],
 'diamonds': ['carat','depth','table','price','x','y','z'],
}

def prepare(name, seed=0):
    df=pd.read_csv(ROOT/f'{name}.csv')
    if name=='diamonds':
        df=df.sample(n=1200, random_state=seed).reset_index(drop=True)
    X=df[SPECS[name]].copy()
    for c in X.columns:
        X[c]=X[c].fillna(X[c].median())
    return X

def kwargs(seed,k,n):
    return dict(n_features=k,n_permutations=16,max_pairs=min(15000,max(1000,n*(n-1)//2)),
                support_engine='vectorized',residual_engine='batched_cd',random_state=seed)

rows=[]; sel_rows=[]
for ds_i,name in enumerate(SPECS):
    Xfull=prepare(name,seed=123+ds_i)
    p=Xfull.shape[1]; k=min(p-1,max(2,int(math.ceil(math.sqrt(p)))))
    ref=GSRSubstitutionSelector(substitution_threshold=0.85,**kwargs(110+ds_i,k,len(Xfull))).fit(Xfull)
    groups=ref.get_substitution_groups()
    rng=np.random.default_rng(9110+ds_i)
    sels=[]
    for r in range(12):
        idx=rng.choice(len(Xfull),size=max(20,int(round(.8*len(Xfull)))),replace=False)
        X=Xfull.iloc[idx]
        est=GSRSelector(**kwargs(2000+100*ds_i+r,k,len(X))).fit(X)
        sel=est.get_support(indices=True).tolist(); sels.append(sel)
        sel_rows.append({'dataset':name,'repeat':r,'selected_indices':','.join(map(str,sel)),
                         'selected_names':','.join(Xfull.columns[j] for j in sel)})
    exact=[]; aware=[]
    for i,j in combinations(range(len(sels)),2):
        exact.append(jaccard(sels[i],sels[j]))
        aware.append(substitution_aware_jaccard(sels[i],sels[j],groups,p))
    rows.append({
        'dataset':name,'n_samples':len(Xfull),'p':p,'k':k,
        'exact_jaccard':float(np.mean(exact)),
        'substitution_aware_jaccard':float(np.mean(aware)),
        'absolute_gain':float(np.mean(aware)-np.mean(exact)),
        'n_grouped_members':int(sum(len(v) for v in groups.values())),
        'n_selected_representatives':int(len(groups)),
        'selected_full':','.join(ref.get_feature_names_out().tolist()),
        'reference_groups':str({str(Xfull.columns[r]):[str(Xfull.columns[j]) for j in m] for r,m in groups.items()}),
    })

pd.DataFrame(rows).to_csv(OUT/'cgs_fresh_structural_panel_summary.csv',index=False)
pd.DataFrame(sel_rows).to_csv(OUT/'cgs_fresh_structural_panel_selections.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
