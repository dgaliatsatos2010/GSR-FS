from __future__ import annotations
from pathlib import Path
import gzip, json, time
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from gsrfs.auxiliary import make_tabular_preprocessor, encoded_k
from gsrfs.downstream import canonical_offline_method_factories

ROOT = Path('/opt/pyvenv/lib/python3.13/site-packages/sklearn/datasets/tests/data/openml')
OUT = Path(__file__).resolve().parent / 'results' / 'v14' / 'openml_fixtures'
OUT.mkdir(parents=True, exist_ok=True)

SPECS = [
    dict(did=3, name='openml3_kr_vs_kp', target='class', ignore=[], cap=900),
    dict(did=40675, name='openml40675_glass2', target='class', ignore=[], cap=None),
    dict(did=62, name='openml62_zoo', target='type', ignore=['animal'], cap=None),
]

def decode_obj(df):
    df=df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode('utf-8') if isinstance(v,(bytes,bytearray)) else v)
    return df

def load_spec(s):
    folder=ROOT/f'id_{s["did"]}'
    f=next(folder.glob('data-v1-dl-*.arff.gz'))
    with gzip.open(f,'rt') as g:
        data,_=arff.loadarff(g)
    df=decode_obj(pd.DataFrame(data))
    # uniform, y-independent cap BEFORE y extraction
    if s['cap'] is not None and len(df)>s['cap']:
        rng=np.random.default_rng(1401)
        idx=np.sort(rng.choice(len(df),size=s['cap'],replace=False))
        df=df.iloc[idx].reset_index(drop=True)
    y=pd.factorize(df[s['target']].astype(str),sort=True)[0].astype(int)
    drop=[s['target']]+s['ignore']
    X=df.drop(columns=[c for c in drop if c in df.columns]).copy()
    return X,y,f

def run_one(s):
    X,y,source_file=load_spec(s)
    methods=canonical_offline_method_factories(gsr_permutations=12,max_pairs=8000)
    rows=[]
    splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
    for fold,(tr,te) in enumerate(splitter.split(X,y)):
        Xtr_raw=X.iloc[tr].copy(); Xte_raw=X.iloc[te].copy(); ytr=y[tr]; yte=y[te]
        pre=make_tabular_preprocessor(Xtr_raw)
        Xtr=pre.fit_transform(Xtr_raw); Xte=pre.transform(Xte_raw)
        scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
        k=encoded_k(Xtr.shape[1])
        for method_name,factory in methods.items():
            selector=factory(k,fold); t0=time.perf_counter()
            try:
                selector.fit(Xtr)
                selected=np.asarray(selector.get_support(indices=True),dtype=int)
                if selected.size==0: raise RuntimeError('empty fixed-k subset')
                clf=LogisticRegression(max_iter=3000,class_weight='balanced')
                clf.fit(Xtr[:,selected],ytr); pred=clf.predict(Xte[:,selected])
                rows.append(dict(dataset=s['name'],openml_dataset_id=s['did'],panel_status='openml_fixture_extension_v014',source_file=str(source_file),seed=0,fold=fold,method=method_name,implementation_source=getattr(selector,'external_source_','native_or_local'),n_samples=len(y),n_raw_features=X.shape[1],n_encoded_features=Xtr.shape[1],n_classes=len(np.unique(y)),k=k,selected=','.join(map(str,selected.tolist())),balanced_accuracy=balanced_accuracy_score(yte,pred),macro_f1=f1_score(yte,pred,average='macro',zero_division=0),accuracy=accuracy_score(yte,pred),selector_fit_seconds=time.perf_counter()-t0,error=''))
            except Exception as exc:
                rows.append(dict(dataset=s['name'],openml_dataset_id=s['did'],panel_status='openml_fixture_extension_v014',source_file=str(source_file),seed=0,fold=fold,method=method_name,implementation_source=('scikit-feature:execution_failed' if 'scikit-feature' in method_name else 'native_or_local'),n_samples=len(y),n_raw_features=X.shape[1],n_encoded_features=Xtr.shape[1],n_classes=len(np.unique(y)),k=k,selected='',balanced_accuracy=np.nan,macro_f1=np.nan,accuracy=np.nan,selector_fit_seconds=time.perf_counter()-t0,error=repr(exc)))
    df=pd.DataFrame(rows); df.to_csv(OUT/f'{s["name"]}_raw.csv',index=False)
    print('\n',s['name']); print(df.groupby('method')[['balanced_accuracy','macro_f1','selector_fit_seconds']].mean().sort_values('balanced_accuracy',ascending=False))
    return df

def main():
    dfs=[run_one(s) for s in SPECS]
    all_df=pd.concat(dfs,ignore_index=True); all_df.to_csv(OUT/'openml_fixture_canonical_raw.csv',index=False)
    summary=all_df.groupby(['dataset','method'],as_index=False)[['balanced_accuracy','macro_f1','accuracy','selector_fit_seconds']].mean()
    summary.to_csv(OUT/'openml_fixture_canonical_summary.csv',index=False)

if __name__=='__main__': main()
