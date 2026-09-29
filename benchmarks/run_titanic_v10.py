from __future__ import annotations

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from gsrfs import GSRSelector, GSRConcordanceSelector
from gsrfs.baselines import MaxVarianceSelector, LaplacianScoreSelector, SPECSelector, MCFSSelector

OUT = Path(__file__).resolve().parent / 'results' / 'v10'
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path('/opt/pyvenv/lib/python3.13/site-packages/gradio/media_assets/data/titanic.csv')


def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 1.0


def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    df = pd.read_csv(DATA)
    y = df['Survived'].to_numpy(dtype=int)
    num = ['Pclass','Age','SibSp','Parch','Fare']
    cat = ['Sex','Embarked']
    Xdf = df[num + cat].copy()
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat),
    ], verbose_feature_names_out=False)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=202610)
    rows=[]
    selections={}
    for fold,(tr,te) in enumerate(cv.split(Xdf,y)):
        Xt = pre.fit_transform(Xdf.iloc[tr])
        Xv = pre.transform(Xdf.iloc[te])
        names = list(pre.get_feature_names_out())
        p = Xt.shape[1]
        k = min(p-1, max(2, int(np.ceil(np.sqrt(p)))))
        factories = {
            'GSR-base': lambda: GSRSelector(n_features=k,n_permutations=12,max_pairs=8000,support_engine='vectorized',random_state=100+fold),
            'GSR-JGC': lambda: GSRConcordanceSelector(n_features=k,tie_tolerance=0.10,concordance_weight=0.0,n_blocks=5,n_permutations=12,max_pairs=8000,support_engine='vectorized',random_state=100+fold),
            'MaxVariance': lambda: MaxVarianceSelector(n_features=k),
            'LaplacianScore-dev': lambda: LaplacianScoreSelector(n_features=k,n_neighbors=7),
            'SPEC-dev': lambda: SPECSelector(n_features=k,gamma=1.0,style=0,standardize=False),
            'MCFS-eigengap-dev': lambda: MCFSSelector(n_features=k,n_clusters='eigengap',n_neighbors=7,alpha=0.003,max_clusters=10,random_state=100+fold),
        }
        for method,factory in factories.items():
            try:
                sel=factory(); sel.fit(Xt)
                idx=sel.get_support(indices=True)
                chosen=tuple(names[i] for i in idx)
                selections.setdefault(method,[]).append(chosen)
                clf=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(max_iter=3000,class_weight='balanced'))])
                clf.fit(Xt[:,idx],y[tr]); pred=clf.predict(Xv[:,idx])
                rows.append({
                    'dataset':'titanic_fresh_v10','fold':fold,'method':method,'k':k,
                    'selected':'|'.join(chosen),
                    'balanced_accuracy':balanced_accuracy_score(y[te],pred),
                    'macro_f1':f1_score(y[te],pred,average='macro',zero_division=0),
                    'error':'',
                })
            except Exception as exc:
                rows.append({
                    'dataset':'titanic_fresh_v10','fold':fold,'method':method,'k':k,
                    'selected':'','balanced_accuracy':np.nan,'macro_f1':np.nan,
                    'error':repr(exc),
                })
    raw=pd.DataFrame(rows)
    raw.to_csv(OUT/'titanic_fresh_downstream_raw.csv',index=False)
    summary=raw.groupby('method')[['balanced_accuracy','macro_f1']].agg(['mean','std','count'])
    summary.to_csv(OUT/'titanic_fresh_downstream_summary.csv')
    stab=[]
    for method,sels in selections.items():
        vals=[jaccard(sels[i],sels[j]) for i,j in combinations(range(len(sels)),2)]
        stab.append({'method':method,'mean_fold_jaccard':float(np.mean(vals)) if vals else np.nan,'n_successful_folds':len(sels),'fold_selections':' || '.join(','.join(x) for x in sels)})
    pd.DataFrame(stab).to_csv(OUT/'titanic_fresh_selection_stability.csv',index=False)
    print(summary)
    print(pd.DataFrame(stab)[['method','mean_fold_jaccard']].sort_values('mean_fold_jaccard',ascending=False))

if __name__=='__main__':
    main()
