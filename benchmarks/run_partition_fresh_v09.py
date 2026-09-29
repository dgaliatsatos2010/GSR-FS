from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from gsrfs import GSRSelector, GSRPartitionSelector

OUT=Path(__file__).resolve().parent/'results'/'v09'; OUT.mkdir(parents=True,exist_ok=True)

def datasets():
    out=[]
    d=sm.datasets.fair.load_pandas().data.copy()
    y=(d.pop('affairs').to_numpy()>0).astype(int); X=d.to_numpy(float); out.append(('Fair_affairs',X,y))
    d=sm.datasets.ccard.load_pandas().data.copy()
    y=d.pop('OWNRENT').to_numpy().astype(int); X=d.to_numpy(float); out.append(('CCard_ownrent',X,y))
    d=sm.datasets.heart.load_pandas().data.copy()
    y=d.pop('censors').to_numpy().astype(int); X=d.to_numpy(float); out.append(('Heart_censor',X,y))
    return out

def k_from_p(p): return min(p-1,max(2,int(np.ceil(np.sqrt(p)))))
rows=[]
for name,X,y in datasets():
    k=k_from_p(X.shape[1]); cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=991)
    for fold,(tr,te) in enumerate(cv.split(X,y)):
        for method in ['GSR-base','GSR-PBGE']:
            if method=='GSR-base': sel=GSRSelector(n_features=k,n_permutations=10,max_pairs=4000,residual_power=.5,random_state=fold+17)
            else: sel=GSRPartitionSelector(n_features=k,partition_weight=.35,n_permutations=10,max_pairs=4000,random_state=fold+17)
            sel.fit(X[tr],y=None)
            idx=sel.get_support(indices=True); scaler=StandardScaler().fit(X[tr][:,idx]); A=scaler.transform(X[tr][:,idx]); B=scaler.transform(X[te][:,idx])
            clf=LogisticRegression(max_iter=2000,class_weight='balanced').fit(A,y[tr]); pred=clf.predict(B)
            rows.append(dict(dataset=name,fold=fold,method=method,k=k,balanced_accuracy=balanced_accuracy_score(y[te],pred),macro_f1=f1_score(y[te],pred,average='macro'),selected=';'.join(map(str,idx))))
df=pd.DataFrame(rows); df.to_csv(OUT/'partition_fresh_downstream_raw.csv',index=False)
smry=df.groupby(['dataset','method'])[['balanced_accuracy','macro_f1']].mean().reset_index(); smry.to_csv(OUT/'partition_fresh_downstream_by_dataset.csv',index=False)
over=df.groupby('method')[['balanced_accuracy','macro_f1']].mean().reset_index(); over.to_csv(OUT/'partition_fresh_downstream_summary.csv',index=False)
print(over.to_string(index=False)); print('\n',smry.to_string(index=False))
