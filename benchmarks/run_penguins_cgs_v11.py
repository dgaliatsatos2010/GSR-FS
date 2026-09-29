from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

from gsrfs import GSRSelector, GSRSubstitutionSelector, substitution_aware_jaccard
from gsrfs.metrics import jaccard

OUT=Path(__file__).resolve().parent/'results'/'v11'; OUT.mkdir(parents=True,exist_ok=True)
DATA='/opt/pyvenv/lib/python3.13/site-packages/plotnine/data/penguins.csv'
FEATURES=['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','year']

def prepare(df):
    X=df[FEATURES].copy()
    for c in FEATURES:
        X[c]=X[c].fillna(X[c].median())
    return X

def kwargs(seed):
    return dict(n_features=3,n_permutations=20,max_pairs=12000,
                support_engine='vectorized',residual_engine='batched_cd',random_state=seed)

df=pd.read_csv(DATA)
Xfull=prepare(df)
ref=GSRSubstitutionSelector(substitution_threshold=0.85,**kwargs(11)).fit(Xfull)
groups=ref.get_substitution_groups()

rng=np.random.default_rng(202611)
sels=[]
rows=[]
for r in range(12):
    idx=rng.choice(len(df),size=int(round(.80*len(df))),replace=False)
    X=prepare(df.iloc[idx])
    est=GSRSelector(**kwargs(100+r)).fit(X)
    sel=est.get_support(indices=True).tolist(); sels.append(sel)
    rows.append({'repeat':r,'selected_indices':','.join(map(str,sel)),
                 'selected_names':','.join(FEATURES[j] for j in sel)})

exact=[]; aware=[]
for i,j in combinations(range(len(sels)),2):
    exact.append(jaccard(sels[i],sels[j]))
    aware.append(substitution_aware_jaccard(sels[i],sels[j],groups,len(FEATURES)))

pd.DataFrame(rows).to_csv(OUT/'penguins_fresh_selections.csv',index=False)
summary=pd.DataFrame([{
    'dataset':'Palmer Penguins (plotnine local copy)',
    'n_samples':len(df),'n_features':len(FEATURES),'k':3,
    'exact_jaccard':float(np.mean(exact)),
    'substitution_aware_jaccard':float(np.mean(aware)),
    'absolute_gain':float(np.mean(aware)-np.mean(exact)),
    'reference_groups':str({FEATURES[r]:[FEATURES[j] for j in m] for r,m in groups.items()}),
    'selected_full':','.join(ref.get_feature_names_out().tolist()),
}])
summary.to_csv(OUT/'penguins_fresh_summary.csv',index=False)
print(summary.to_string(index=False))
print(ref.get_group_report().to_string(index=False))
